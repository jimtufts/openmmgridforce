"""Isolate the source of the ~4.5 kJ/mol delta between PAIRWISE ligand-self
and pure GRID mode on the same receptor+ligand system.

Prints the ligand-self / total / SA-inclusive numbers for both paths at
SA=off and SA=on, so we can see whether the delta is grid interpolation
error only or has a hidden SA contribution.
"""
import numpy as np
import openmm as mm
from openmm import unit
import gridforceplugin as gfp


np.random.seed(42)
n_rec, n_lig = 40, 12
rec_pos = np.random.randn(n_rec, 3) * 0.4
lig_pos = np.random.randn(n_lig, 3) * 0.15 + np.array([0.9, 0.0, 0.0])
rec_q = np.random.randn(n_rec) * 0.3
lig_q = np.random.randn(n_lig) * 0.3
rec_r = 0.12 + np.random.rand(n_rec) * 0.06
lig_r = 0.12 + np.random.rand(n_lig) * 0.06
rec_s = 0.7 + np.random.rand(n_rec) * 0.2
lig_s = 0.7 + np.random.rand(n_lig) * 0.2

all_pos = np.vstack([rec_pos, lig_pos])
lo = all_pos.min(axis=0) - 0.6
hi = all_pos.max(axis=0) + 0.6
sp = 0.15
counts = tuple(int(np.ceil((hi[d] - lo[d]) / sp)) + 1 for d in range(3))


def _context(system, spec_name, spec_props):
    plat = mm.Platform.getPlatformByName(spec_name)
    integ = mm.VerletIntegrator(0.001)
    return mm.Context(system, integ, plat, spec_props), integ


def build_pairwise(include_sa):
    system = mm.System()
    for _ in range(n_lig):
        system.addParticle(12.0)
    f = gfp.IsolatedGBSAForce()
    f.setGBMethod(gfp.IsolatedGBSAForce.OBC_II)
    f.setSoluteDielectric(1.0); f.setSolventDielectric(78.5)
    f.setIncludeSurfaceArea(include_sa)
    f.setReceptorMode(gfp.IsolatedGBSAForce.PAIRWISE)
    f.setNumAtoms(n_lig)
    for i in range(n_lig):
        f.setAtomParameters(i, float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
    f.setNumReceptorAtoms(n_rec)
    for i in range(n_rec):
        f.setReceptorAtomParameters(i, float(rec_q[i]), float(rec_r[i]), float(rec_s[i]))
    f.setReceptorPositions(rec_pos.flatten().tolist())
    f.addParticleGroup("lig", list(range(n_lig)))
    system.addForce(f); return system, f


def compute_receptor_baseline():
    R = rec_r.astype(np.float64); S = rec_s.astype(np.float64)
    p = rec_pos.astype(np.float64)
    OFFSET = 0.009
    R_off = R - OFFSET
    hct = np.zeros(n_rec)
    for i in range(n_rec):
        for j in range(n_rec):
            if i == j: continue
            r = float(np.linalg.norm(p[i] - p[j]))
            Sj = (R[j] - OFFSET) * S[j]
            r_plus_Sj = r + Sj
            if R_off[i] >= r_plus_Sj: continue
            r_minus_Sj = abs(r - Sj)
            l = 1.0/R_off[i] if R_off[i] > r_minus_Sj else 1.0/r_minus_Sj
            u = 1.0 / r_plus_Sj
            l2, u2 = l*l, u*u
            term = (l - u + 0.25*r*(u2 - l2) + 0.5*(1.0/r)*np.log(u/l)
                    + 0.25*Sj*Sj*(1.0/r)*(l2 - u2))
            if R_off[i] < (Sj - r):
                term += 2.0 * (1.0/R_off[i] - l)
            hct[i] += term
    A, B, G = 1.0, 0.8, 4.85
    psi = 0.5 * R_off * hct
    tanh_val = np.tanh(A*psi - B*psi**2 + G*psi**3)
    denom = 1.0/R_off - tanh_val / R
    return np.minimum(np.where(denom > 0, 1.0/denom, R), 50.0)


def autogen_grid():
    f_gen = gfp.GBSAGridForce()
    f_gen.setNumAtoms(1)
    f_gen.setAtomParameters(0, 0.0, float(lig_r[0]), float(lig_s[0]))
    f_gen.setInterpolationMethod(0)
    f_gen.setAutoGenerateGrid(True); f_gen.setUseKDEGeneration(True)
    f_gen.setReceptorPositions(rec_pos.flatten().tolist())
    f_gen.setReceptorRadii(rec_r.tolist())
    f_gen.setReceptorScaleFactors(rec_s.tolist())
    f_gen.setGridOrigin(float(lo[0]), float(lo[1]), float(lo[2]))
    f_gen.setGridCounts(int(counts[0]), int(counts[1]), int(counts[2]))
    f_gen.setGridSpacing(float(sp))
    f_gen.setProbeRadius(0.14); f_gen.setRThresholds([0.12, 0.16])
    f_gen.setParticles([0]); f_gen.addParticleGroup('bootstrap', [0])
    s = mm.System(); s.addParticle(12.0); s.addForce(f_gen)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName('CUDA'),
                     {'Precision': 'double'})
    ctx.setPositions([[0.0, 0.0, 0.0]] * unit.nanometer)
    ctx.getState(getEnergy=True)
    cg = f_gen.getDesolvationGrid()
    del ctx
    return cg


cg = autogen_grid()
R_rec_baseline = compute_receptor_baseline()


def build_grid(include_sa):
    system = mm.System()
    for _ in range(n_lig):
        system.addParticle(12.0)
    f = gfp.IsolatedGBSAForce()
    f.setGBMethod(gfp.IsolatedGBSAForce.OBC_II)
    f.setSoluteDielectric(1.0); f.setSolventDielectric(78.5)
    f.setIncludeSurfaceArea(include_sa)
    f.setReceptorMode(gfp.IsolatedGBSAForce.GRID)
    f.setNumAtoms(n_lig)
    for i in range(n_lig):
        f.setAtomParameters(i, float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
    f.setDesolvationGrid(cg); f.setInterpolationMethod(0)
    f.setNumReceptorAtoms(n_rec)
    for i in range(n_rec):
        f.setReceptorAtomParameters(i, float(rec_q[i]), float(rec_r[i]), float(rec_s[i]))
    f.setReceptorPositions(rec_pos.flatten().tolist())
    f.setReceptorBornRadiiBaseline([float(x) for x in R_rec_baseline])
    f.addParticleGroup("lig", list(range(n_lig)))
    system.addForce(f); return system, f


def eval_(system, force):
    ctx, integ = _context(system, 'CUDA', {'Precision': 'double'})
    ctx.setPositions(lig_pos * unit.nanometer)
    E = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    parts = dict(total=E,
                 ligself=force.getGroupLigandSelfEnergy(0),
                 cross=force.getGroupCrossTermEnergy(0),
                 rec_contrib=force.getGroupReceptorContribution(0),
                 rec_desolv=force.getGroupReceptorDesolvation(0))
    del ctx, integ
    return parts


for label, include_sa in [('SA=off', False), ('SA=on', True)]:
    print(f"\n=== {label} ===")
    p_sys, p_f = build_pairwise(include_sa)
    g_sys, g_f = build_grid(include_sa)
    p = eval_(p_sys, p_f)
    g = eval_(g_sys, g_f)
    for k in ('total', 'ligself', 'cross', 'rec_contrib', 'rec_desolv'):
        print(f"  {k:12s}  PAIRWISE={p[k]:+9.3f}   GRID={g[k]:+9.3f}   Δ(GRID−P)={g[k]-p[k]:+7.3f}")
    d_lig = g['total'] - p['ligself']
    print(f"  Δ(GRID.total − PAIRWISE.ligself) = {d_lig:+7.3f}")
