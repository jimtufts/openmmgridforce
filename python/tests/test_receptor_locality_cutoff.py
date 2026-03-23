#!/usr/bin/env python
"""
Receptor Locality Cutoff Validation for IsolatedGBSAForce PAIRWISE Mode

Tests the receptorLocalityCutoff optimization by comparing energy with
various cutoffs against the ground truth (no cutoff).

The optimization keeps ALL receptor atoms but only updates Born radii for
receptor atoms within the locality cutoff of the ligand. Distant atoms
keep their reference (ligand-free) Born radii.
"""

import functools
import numpy as np
import sys
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
    """Extract GBSA parameters from AMBER files using OpenMM."""
    prmtop = AmberPrmtopFile(prmtop_file)
    inpcrd = AmberInpcrdFile(inpcrd_file)
    system = prmtop.createSystem(
        nonbondedMethod=mm.app.NoCutoff,
        implicitSolvent=mm.app.OBC2
    )
    gb = nb = None
    for f in system.getForces():
        if isinstance(f, mm.GBSAOBCForce):
            gb = f
        if isinstance(f, mm.NonbondedForce):
            nb = f
    n = system.getNumParticles()
    positions = np.array(inpcrd.positions.value_in_unit(unit.nanometer))
    charges = np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                       for i in range(n)])
    radii = np.array([gb.getParticleParameters(i)[1].value_in_unit(unit.nanometer)
                     for i in range(n)])
    scales = np.array([gb.getParticleParameters(i)[2] for i in range(n)])
    return positions, charges, radii, scales


def compute_pairwise_gbsa(lig_pos, lig_charges, lig_radii, lig_scales,
                           rec_pos, rec_charges, rec_radii, rec_scales,
                           locality_cutoff=-1.0):
    """
    Compute PAIRWISE GBSA energy using IsolatedGBSAForce.
    If locality_cutoff > 0, use receptor locality optimization.
    """
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

    if locality_cutoff > 0:
        force.setReceptorLocalityCutoff(locality_cutoff)

    force.addParticleGroup("ligand", list(range(n_lig)))

    system.addForce(force)

    platform = mm.Platform.getPlatformByName('CUDA')
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, platform)
    context.setPositions(lig_pos * unit.nanometer)

    state = context.getState(getEnergy=True)
    total = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    result = {
        'total_energy': total,
        'ligand_self': force.getGroupLigandSelfEnergy(0),
        'receptor_desolvation': force.getGroupReceptorDesolvation(0),
        'cross_term': force.getGroupCrossTermEnergy(0),
        'born_radii': np.array(force.getGroupBornRadii(0)),
    }

    del context, integrator
    return result


def perturb_ligand_pose(positions, translation_mag=0.2, rotation=True, rng=None):
    """Apply random translation and rotation to ligand positions."""
    if rng is None:
        rng = np.random.default_rng()
    translation = rng.normal(0, translation_mag, size=3)
    new_positions = positions + translation
    if rotation:
        com = new_positions.mean(axis=0)
        centered = new_positions - com
        angle = rng.normal(0, 0.3)
        axis = rng.normal(0, 1, size=3)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        new_positions = (R @ centered.T).T + com
    return new_positions


def main():
    print("Loading receptor and ligand parameters...")
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
    print(f"Receptor: {n_rec} atoms")
    print(f"Ligand:   {n_lig} atoms")

    cutoffs = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

    poses = [("crystal", lig_pos)]
    rng = np.random.default_rng(42)
    for p in range(5):
        poses.append((f"perturbed_{p+1}",
                      perturb_ligand_pose(lig_pos, translation_mag=0.2, rng=rng)))

    all_results = {}

    for pose_label, pose_pos in poses:
        print(f"\n{'='*80}")
        print(f"{pose_label.upper()}")
        print(f"{'='*80}")

        # Ground truth: no locality cutoff
        print(f"  Ground truth (no locality cutoff)...")
        t0 = time.time()
        truth = compute_pairwise_gbsa(
            pose_pos, lig_charges, lig_radii, lig_scales,
            rec_pos, rec_charges, rec_radii, rec_scales,
            locality_cutoff=-1.0
        )
        t_gt = time.time() - t0
        print(f"    Total: {truth['total_energy']:.4f} kJ/mol  ({t_gt:.2f}s)")
        print(f"    Ligand self: {truth['ligand_self']:.4f}")
        print(f"    Rec desolv:  {truth['receptor_desolvation']:.4f}")
        print(f"    Cross-term:  {truth['cross_term']:.4f}")

        pose_results = []
        for cutoff in cutoffs:
            t0 = time.time()
            approx = compute_pairwise_gbsa(
                pose_pos, lig_charges, lig_radii, lig_scales,
                rec_pos, rec_charges, rec_radii, rec_scales,
                locality_cutoff=cutoff
            )
            t_approx = time.time() - t0

            total_err = approx['total_energy'] - truth['total_energy']
            desolv_err = approx['receptor_desolvation'] - truth['receptor_desolvation']
            cross_err = approx['cross_term'] - truth['cross_term']

            # Ligand Born radii comparison
            lig_born_diff = approx['born_radii'] - truth['born_radii']
            lig_born_rmsd = np.sqrt(np.mean(lig_born_diff**2))

            result = {
                'cutoff': cutoff,
                'total_error': total_err,
                'desolv_error': desolv_err,
                'cross_error': cross_err,
                'lig_born_rmsd': lig_born_rmsd,
            }
            pose_results.append(result)

            print(f"  Cutoff {cutoff:.1f} nm | "
                  f"Total err: {total_err:+8.4f} | Desolv err: {desolv_err:+8.4f} | "
                  f"Cross err: {cross_err:+8.4f} | "
                  f"LigBorn RMSD: {lig_born_rmsd:.6f}  ({t_approx:.2f}s)")

        all_results[pose_label] = pose_results

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Maximum absolute errors across all poses (kJ/mol)")
    print(f"{'='*80}")
    print(f"{'Cutoff':>8s}  "
          f"{'Total':>10s}  {'Desolv':>10s}  {'Cross':>10s}  "
          f"{'LigBorn':>10s}")
    print("-" * 55)

    for ci, cutoff in enumerate(cutoffs):
        total_errs, desolv_errs, cross_errs = [], [], []
        lig_born_rmsds = []

        for pose_results in all_results.values():
            r = pose_results[ci]
            total_errs.append(abs(r['total_error']))
            desolv_errs.append(abs(r['desolv_error']))
            cross_errs.append(abs(r['cross_error']))
            lig_born_rmsds.append(r['lig_born_rmsd'])

        print(f"{cutoff:8.1f}  "
              f"{max(total_errs):10.4f}  {max(desolv_errs):10.4f}  "
              f"{max(cross_errs):10.4f}  "
              f"{max(lig_born_rmsds):10.6f}")

    print("\nNote: Locality cutoff prunes receptor atoms from HCT on ligand,")
    print("receptor desolvation, and forces. Cross-term uses mixed Born radii.")
    print("Target: Total error < 1-4 kJ/mol")

    # ============================================================
    # Timing benchmark: repeated evaluations with same Context
    # ============================================================
    n_evals = 400

    print(f"\n{'='*80}")
    print(f"TIMING BENCHMARK (crystal pose, {n_evals} evaluations each)")
    print(f"{'='*80}")

    for locality in [-1.0, 2.0, 2.5]:
        label = "no cutoff" if locality < 0 else f"{locality:.1f} nm"

        # Create context once
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

        platform = mm.Platform.getPlatformByName('CUDA')
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator, platform)
        context.setPositions(lig_pos * unit.nanometer)

        # Warmup
        state = context.getState(getEnergy=True)

        # Benchmark: re-evaluate with small position perturbations
        rng2 = np.random.default_rng(123)
        times = []
        for _ in range(n_evals):
            perturbed = lig_pos + rng2.normal(0, 0.01, size=lig_pos.shape)
            context.setPositions(perturbed * unit.nanometer)
            t0 = time.time()
            state = context.getState(getEnergy=True, getForces=True)
            times.append(time.time() - t0)

        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        print(f"  Locality {label:>12s}: {avg_ms:.1f} +/- {std_ms:.1f} ms/eval")

        del context, integrator


if __name__ == '__main__':
    main()
