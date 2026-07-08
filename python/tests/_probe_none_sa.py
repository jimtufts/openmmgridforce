import numpy as np, openmm as mm
from openmm import unit
import gridforceplugin as gfp

np.random.seed(42)
n_lig = 12
lig_pos = np.random.randn(n_lig, 3) * 0.15 + np.array([0.9, 0.0, 0.0])
lig_q = np.random.randn(n_lig) * 0.3
lig_r = 0.12 + np.random.rand(n_lig) * 0.06
lig_s = 0.7 + np.random.rand(n_lig) * 0.2

def build_none(sa):
    sy = mm.System()
    for _ in range(n_lig): sy.addParticle(12.0)
    f = gfp.IsolatedGBSAForce()
    f.setGBMethod(gfp.IsolatedGBSAForce.OBC_II)
    f.setSoluteDielectric(1.0); f.setSolventDielectric(78.5)
    f.setIncludeSurfaceArea(sa)
    f.setReceptorMode(gfp.IsolatedGBSAForce.NONE)
    f.setNumAtoms(n_lig)
    for i in range(n_lig):
        f.setAtomParameters(i, float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
    f.addParticleGroup('lig', list(range(n_lig)))
    sy.addForce(f); return sy, f

for plat_name, props in [('Reference', {}), ('CUDA', {'Precision': 'double'})]:
    print(f'\n== {plat_name} NONE ==')
    for sa in (False, True):
        sy, f = build_none(sa)
        plat = mm.Platform.getPlatformByName(plat_name)
        ctx = mm.Context(sy, mm.VerletIntegrator(0.001), plat, props)
        ctx.setPositions(lig_pos * unit.nanometer)
        E = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        ls = f.getGroupLigandSelfEnergy(0)
        print(f'  SA={"on " if sa else "off"}  total={E:+.4f}  ligself={ls:+.4f}  Δ(total-ligself)={E-ls:+.4f}')
        del ctx
