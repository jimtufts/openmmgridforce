#!/usr/bin/env python
"""
Profile IsolatedGBSAForce PAIRWISE mode on both CUDA and Reference platforms.
"""

import functools
import numpy as np
import os
import time

print = functools.partial(print, flush=True)

from openmm.app import AmberPrmtopFile, AmberInpcrdFile
from openmm import unit
import openmm as mm
import gridforceplugin as gf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRMTOP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'prmtopcrd')


def get_gbsa_params(prmtop_file, inpcrd_file):
    prmtop = AmberPrmtopFile(prmtop_file)
    inpcrd = AmberInpcrdFile(inpcrd_file)
    system = prmtop.createSystem(nonbondedMethod=mm.app.NoCutoff, implicitSolvent=mm.app.OBC2)
    gb = nb = None
    for f in system.getForces():
        if isinstance(f, mm.GBSAOBCForce): gb = f
        if isinstance(f, mm.NonbondedForce): nb = f
    n = system.getNumParticles()
    positions = np.array(inpcrd.positions.value_in_unit(unit.nanometer))
    charges = np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in range(n)])
    radii = np.array([gb.getParticleParameters(i)[1].value_in_unit(unit.nanometer) for i in range(n)])
    scales = np.array([gb.getParticleParameters(i)[2] for i in range(n)])
    return positions, charges, radii, scales


def bench(lig_pos, lig_charges, lig_radii, lig_scales,
          rec_pos, rec_charges, rec_radii, rec_scales,
          platform, locality=-1.0, n_evals=50):
    n_lig = len(lig_pos)
    n_rec = len(rec_pos)

    system = mm.System()
    for _ in range(n_lig):
        system.addParticle(12.0)

    force = gf.IsolatedGBSAForce()
    force.setGBMethod(gf.IsolatedGBSAForce.OBC_II)
    force.setSoluteDielectric(1.0)
    force.setSolventDielectric(78.5)
    force.setIncludeSurfaceArea(False)
    force.setNumAtoms(n_lig)
    for i in range(n_lig):
        force.setAtomParameters(i, float(lig_charges[i]),
                                float(lig_radii[i]), float(lig_scales[i]))
    force.setReceptorMode(gf.IsolatedGBSAForce.PAIRWISE)
    force.setNumReceptorAtoms(n_rec)
    for i in range(n_rec):
        force.setReceptorAtomParameters(i, float(rec_charges[i]),
                                        float(rec_radii[i]), float(rec_scales[i]))
    force.setReceptorPositions(rec_pos.flatten().tolist())
    if locality > 0:
        force.setReceptorLocalityCutoff(locality)
    force.addParticleGroup("ligand", list(range(n_lig)))
    system.addForce(force)

    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, platform)
    context.setPositions(lig_pos * unit.nanometer)

    # Warmup
    state = context.getState(getEnergy=True, getForces=True)

    rng = np.random.default_rng(42)
    times = []
    for _ in range(n_evals):
        pos = lig_pos + rng.normal(0, 0.01, size=lig_pos.shape)
        context.setPositions(pos * unit.nanometer)
        t0 = time.time()
        state = context.getState(getEnergy=True, getForces=True)
        times.append(time.time() - t0)

    del context, integrator
    return np.mean(times) * 1000, np.std(times) * 1000


def main():
    print("Loading parameters...")
    rec_pos, rec_charges, rec_radii, rec_scales = get_gbsa_params(
        os.path.join(PRMTOP_DIR, 'receptor.prmtop'),
        os.path.join(PRMTOP_DIR, 'receptor.trans.inpcrd')
    )
    lig_pos, lig_charges, lig_radii, lig_scales = get_gbsa_params(
        os.path.join(PRMTOP_DIR, 'ligand.prmtop'),
        os.path.join(PRMTOP_DIR, 'ligand.trans.inpcrd')
    )
    n_rec = len(rec_pos)
    n_lig = len(lig_pos)
    print(f"Receptor: {n_rec} atoms, Ligand: {n_lig} atoms")

    cuda_platform = mm.Platform.getPlatformByName('CUDA')
    ref_platform = mm.Platform.getPlatformByName('Reference')

    # Get GPU info
    s = mm.System(); s.addParticle(1.0)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(s, integ, cuda_platform)
    try:
        print(f"GPU: {cuda_platform.getPropertyValue(ctx, 'DeviceName')}")
    except:
        pass
    del ctx, integ

    print(f"\n{'Platform':<12s} {'RecSize':>8s} {'Locality':>10s} {'ms/eval':>10s} {'Speedup':>8s}")
    print("-" * 55)

    # Reference platform (fewer evals since it's slow)
    for rec_n, n_evals in [(2000, 5), (9133, 2)]:
        rec_sub = rec_pos[:rec_n]
        rec_sub_c = rec_charges[:rec_n]
        rec_sub_r = rec_radii[:rec_n]
        rec_sub_s = rec_scales[:rec_n]

        t_ref_off, _ = bench(lig_pos, lig_charges, lig_radii, lig_scales,
                             rec_sub, rec_sub_c, rec_sub_r, rec_sub_s,
                             ref_platform, locality=-1.0, n_evals=n_evals)

        t_ref_on, _ = bench(lig_pos, lig_charges, lig_radii, lig_scales,
                            rec_sub, rec_sub_c, rec_sub_r, rec_sub_s,
                            ref_platform, locality=2.0, n_evals=n_evals)

        speedup = t_ref_off / t_ref_on if t_ref_on > 0 else 0
        print(f"{'Reference':<12s} {rec_n:>8d} {'off':>10s} {t_ref_off:>10.1f}")
        print(f"{'Reference':<12s} {rec_n:>8d} {'2.0 nm':>10s} {t_ref_on:>10.1f} {speedup:>7.2f}x")

    # CUDA platform
    for rec_n in [500, 2000, 9133]:
        rec_sub = rec_pos[:rec_n]
        rec_sub_c = rec_charges[:rec_n]
        rec_sub_r = rec_radii[:rec_n]
        rec_sub_s = rec_scales[:rec_n]

        t_cuda_off, _ = bench(lig_pos, lig_charges, lig_radii, lig_scales,
                              rec_sub, rec_sub_c, rec_sub_r, rec_sub_s,
                              cuda_platform, locality=-1.0, n_evals=100)

        t_cuda_on, _ = bench(lig_pos, lig_charges, lig_radii, lig_scales,
                             rec_sub, rec_sub_c, rec_sub_r, rec_sub_s,
                             cuda_platform, locality=2.0, n_evals=100)

        speedup = t_cuda_off / t_cuda_on if t_cuda_on > 0 else 0
        print(f"{'CUDA':<12s} {rec_n:>8d} {'off':>10s} {t_cuda_off:>10.1f}")
        print(f"{'CUDA':<12s} {rec_n:>8d} {'2.0 nm':>10s} {t_cuda_on:>10.1f} {speedup:>7.2f}x")


if __name__ == '__main__':
    main()
