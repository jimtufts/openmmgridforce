/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CUDA kernels for the GRID-mode receptor add-ons of IsolatedGBSAForce:
 * the radius-sliced cross term (CROSS_RADIUS_GRID) and the linear-response
 * mirror term (MIRROR_LINEAR_GRID).
 *
 * Both split into a gridded far field and a pairwise near shell joined by a
 * smootherstep switch, so the lattice never has to represent a near-contact
 * cusp. The near shells run off a uniform cell list over the pocket atoms,
 * built once because the receptor is rigid; scanning every pocket atom would
 * cost as much as the exact sum these replace.
 *
 * Pocket-local indexing throughout: cellAtoms holds indices into the pocket
 * arrays, not the full receptor, so the per-group scratch buffers scale with
 * the pocket rather than the protein.
 *
 * Cross term, per ligand atom i:
 *   E = pref q_i [ (1-w) Phi_k(r_i) + w Phi_{k+1}(r_i) ]
 *     + pref q_i sum_{j near} q_j [ 1/f_GB(r,R_i,R'_j) - S(r) far_j ]
 * where R'_j is the receptor Born radius re-solved from the ligand-induced
 * descreening, and far_j is exactly what the two bracketing slices already
 * contributed for that pair at apo radii.
 *
 * Mirror term, per ligand atom i:
 *   E = SCALE [ Psi_b(r_i) + sum_{j near} w_j H(r, rho_off_j, s_b) (1 - S(r)) ]
 * -------------------------------------------------------------------------- */

// Guarded: this source is concatenated with isolatedGBSA.cu, which defines
// the same physical constants.
#ifndef DIELECTRIC_OFFSET
#define DIELECTRIC_OFFSET 0.009f
#endif
#ifndef OBC_ALPHA
#define OBC_ALPHA 1.0f
#define OBC_BETA 0.8f
#define OBC_GAMMA 4.85f
#endif

// Distance floor matching the Reference cross term (0.1 nm).
#define MIN_CROSS_R2 0.01f

// =============================================================================
// Shared device helpers
// =============================================================================

__device__ static inline real fieldSwitch(real r, real rOn, real rOff) {
    if (rOff <= rOn) return (real) 1;
    if (r <= rOn) return (real) 0;
    if (r >= rOff) return (real) 1;
    real t = (r - rOn) / (rOff - rOn);
    return t * t * t * (t * (t * (real) 6 - (real) 15) + (real) 10);
}

__device__ static inline real fieldSwitchDeriv(real r, real rOn, real rOff) {
    if (rOff <= rOn || r <= rOn || r >= rOff) return (real) 0;
    real inv = (real) 1 / (rOff - rOn);
    real t = (r - rOn) * inv;
    return (real) 30 * t * t * (t - (real) 1) * (t - (real) 1) * inv;
}

/** HCT integral, plugin convention (psi = 0.5 * R_off * hct). */
__device__ static inline real hctTerm(real r, real Rp_off, real S) {
    real rp = r + S;
    if (Rp_off >= rp) return (real) 0;
    real rm = fabs(r - S);
    real l = (Rp_off > rm) ? ((real) 1 / Rp_off) : ((real) 1 / rm);
    real u = (real) 1 / rp;
    real l2 = l * l, u2 = u * u, rinv = (real) 1 / r;
    real term = l - u + (real) 0.25 * r * (u2 - l2)
              + (real) 0.5 * rinv * log(u / l)
              + (real) 0.25 * S * S * rinv * (l2 - u2);
    if (Rp_off < (S - r))
        term += (real) 2 * ((real) 1 / Rp_off - l);
    return term;
}

/** d(hctTerm)/dr. */
__device__ static inline real hctTermDeriv(real r, real Rp_off, real S) {
    real rp = r + S;
    if (Rp_off >= rp) return (real) 0;
    real rm = fabs(r - S);
    bool lFixed = (Rp_off > rm);
    real l = lFixed ? ((real) 1 / Rp_off) : ((real) 1 / rm);
    real u = (real) 1 / rp;
    real l2 = l * l, u2 = u * u;
    real rinv = (real) 1 / r, rinv2 = rinv * rinv;
    real du = -u2;
    real dl = lFixed ? (real) 0 : ((r > S) ? -l2 : l2);
    real d = dl - du
           + (real) 0.25 * (u2 - l2)
           + (real) 0.25 * r * ((real) 2 * u * du - (real) 2 * l * dl)
           - (real) 0.5 * rinv2 * log(u / l)
           + (real) 0.5 * rinv * (du / u - dl / l)
           - (real) 0.25 * S * S * rinv2 * (l2 - u2)
           + (real) 0.25 * S * S * rinv * ((real) 2 * l * dl - (real) 2 * u * du);
    return d;
}

__device__ static inline real bornFromHCT(real rho, real hct, int useOBC) {
    real R_off = rho - DIELECTRIC_OFFSET;
    if (R_off <= (real) 0) return (real) 500;
    real inner;
    if (!useOBC) {
        inner = (real) 1 / R_off - (real) 0.5 * R_off * hct;
    } else {
        real psi = (real) 0.5 * R_off * hct;
        real t = tanh(OBC_ALPHA * psi - OBC_BETA * psi * psi
                      + OBC_GAMMA * psi * psi * psi);
        inner = (real) 1 / R_off - t / rho;
    }
    return (inner > (real) 0) ? ((real) 1 / inner) : (real) 500;
}

__device__ static inline real dBornDHCT(real rho, real hct, real born, int useOBC) {
    real R_off = rho - DIELECTRIC_OFFSET;
    if (R_off <= (real) 0) return (real) 0;
    if (!useOBC) return (real) 0.5 * R_off * born * born;
    real psi = (real) 0.5 * R_off * hct;
    real t = tanh(OBC_ALPHA * psi - OBC_BETA * psi * psi
                  + OBC_GAMMA * psi * psi * psi);
    real dt = ((real) 1 - t * t)
            * (OBC_ALPHA - (real) 2 * OBC_BETA * psi
               + (real) 3 * OBC_GAMMA * psi * psi);
    return born * born * dt * (real) 0.5 * R_off / rho;
}

__device__ static inline real invFgb(real r2, real Ri, real Rj) {
    real D = Ri * Rj;
    return rsqrt(r2 + D * exp(-r2 / ((real) 4 * D)));
}

/** Cubic B-spline weights and their derivatives for a fractional offset. */
__device__ static inline void cubicWeights(real t, real* w, real* dw) {
    real t2 = t * t, t3 = t2 * t;
    w[0] = ((real) 1 - (real) 3 * t + (real) 3 * t2 - t3) / (real) 6;
    w[1] = ((real) 4 - (real) 6 * t2 + (real) 3 * t3) / (real) 6;
    w[2] = ((real) 1 + (real) 3 * t + (real) 3 * t2 - (real) 3 * t3) / (real) 6;
    w[3] = t3 / (real) 6;
    dw[0] = (-(real) 3 + (real) 6 * t - (real) 3 * t2) / (real) 6;
    dw[1] = (-(real) 12 * t + (real) 9 * t2) / (real) 6;
    dw[2] = ((real) 3 + (real) 6 * t - (real) 9 * t2) / (real) 6;
    dw[3] = (real) 3 * t2 / (real) 6;
}

/**
 * Sample one slice of a SolvationFieldGrid, matching
 * SolvationFields::interpolateField. Returns 0 with a zero gradient outside
 * the interpolable region.
 */
__device__ static inline real sampleField(
        const float* __restrict__ slice, int nx, int ny, int nz,
        real ox, real oy, real oz, real spacing, int method,
        real px, real py, real pz, int wantGrad,
        real* gx, real* gy, real* gz) {

    *gx = (real) 0; *gy = (real) 0; *gz = (real) 0;
    real invSp = (real) 1 / spacing;
    real fx = (px - ox) * invSp;
    real fy = (py - oy) * invSp;
    real fz = (pz - oz) * invSp;
    int ix = (int) floor(fx);
    int iy = (int) floor(fy);
    int iz = (int) floor(fz);
    if (ix < 0 || ix >= nx - 1 || iy < 0 || iy >= ny - 1 || iz < 0 || iz >= nz - 1)
        return (real) 0;

    int nyz = ny * nz;
    real tx = fx - ix, ty = fy - iy, tz = fz - iz;

    if (method == 1) {
        real wx[4], wy[4], wz[4], dwx[4], dwy[4], dwz[4];
        cubicWeights(tx, wx, dwx);
        cubicWeights(ty, wy, dwy);
        cubicWeights(tz, wz, dwz);
        real val = 0, dvx = 0, dvy = 0, dvz = 0;
        for (int a = 0; a < 4; a++) {
            int cx = min(max(ix - 1 + a, 0), nx - 1);
            for (int b = 0; b < 4; b++) {
                int cy = min(max(iy - 1 + b, 0), ny - 1);
                for (int c = 0; c < 4; c++) {
                    int cz = min(max(iz - 1 + c, 0), nz - 1);
                    real v = slice[cx * nyz + cy * nz + cz];
                    val += wx[a] * wy[b] * wz[c] * v;
                    if (wantGrad) {
                        dvx += dwx[a] * wy[b] * wz[c] * v;
                        dvy += wx[a] * dwy[b] * wz[c] * v;
                        dvz += wx[a] * wy[b] * dwz[c] * v;
                    }
                }
            }
        }
        if (wantGrad) { *gx = dvx * invSp; *gy = dvy * invSp; *gz = dvz * invSp; }
        return val;
    }

    int base = ix * nyz + iy * nz + iz;
    real c000 = slice[base];
    real c001 = slice[base + 1];
    real c010 = slice[base + nz];
    real c011 = slice[base + nz + 1];
    real c100 = slice[base + nyz];
    real c101 = slice[base + nyz + 1];
    real c110 = slice[base + nyz + nz];
    real c111 = slice[base + nyz + nz + 1];
    real mx = (real) 1 - tx, my = (real) 1 - ty, mz = (real) 1 - tz;
    real val = mx * my * mz * c000 + mx * my * tz * c001
             + mx * ty * mz * c010 + mx * ty * tz * c011
             + tx * my * mz * c100 + tx * my * tz * c101
             + tx * ty * mz * c110 + tx * ty * tz * c111;
    if (wantGrad) {
        *gx = invSp * (my * mz * (c100 - c000) + my * tz * (c101 - c001)
                     + ty * mz * (c110 - c010) + ty * tz * (c111 - c011));
        *gy = invSp * (mx * mz * (c010 - c000) + mx * tz * (c011 - c001)
                     + tx * mz * (c110 - c100) + tx * tz * (c111 - c101));
        *gz = invSp * (mx * my * (c001 - c000) + mx * ty * (c011 - c010)
                     + tx * my * (c101 - c100) + tx * ty * (c111 - c110));
    }
    return val;
}

/**
 * Which group a flat ligand-atom index belongs to. Every group holds one copy
 * of the template, so this is a division rather than a scan over groups.
 */
__device__ static inline int findGroup(int numGroups, int templateNumAtoms,
                                       int idx, int* atomInGroup) {
    int g = idx / templateNumAtoms;
    if (g >= numGroups) return -1;
    *atomInGroup = idx - g * templateNumAtoms;
    return g;
}


/**
 * Sample two slices at once. The bracketing cross-field slices share stencil
 * indices and basis weights, so walking the 4x4x4 neighbourhood twice doubles
 * the index math and the clamping for nothing.
 */
__device__ static inline void sampleFieldPair(
        const float* __restrict__ sliceA, const float* __restrict__ sliceB,
        int nx, int ny, int nz, real ox, real oy, real oz, real spacing,
        int method, real px, real py, real pz,
        real* vA, real* gAx, real* gAy, real* gAz,
        real* vB, real* gBx, real* gBy, real* gBz) {

    *vA = *gAx = *gAy = *gAz = (real) 0;
    *vB = *gBx = *gBy = *gBz = (real) 0;
    real invSp = (real) 1 / spacing;
    real fx = (px - ox) * invSp, fy = (py - oy) * invSp, fz = (pz - oz) * invSp;
    int ix = (int) floor(fx), iy = (int) floor(fy), iz = (int) floor(fz);
    if (ix < 0 || ix >= nx - 1 || iy < 0 || iy >= ny - 1 || iz < 0 || iz >= nz - 1)
        return;

    int nyz = ny * nz;
    real tx = fx - ix, ty = fy - iy, tz = fz - iz;

    if (method == 1) {
        real wx[4], wy[4], wz[4], dwx[4], dwy[4], dwz[4];
        cubicWeights(tx, wx, dwx);
        cubicWeights(ty, wy, dwy);
        cubicWeights(tz, wz, dwz);
        real a = 0, ax = 0, ay = 0, az = 0, b = 0, bx = 0, by = 0, bz = 0;
        for (int u = 0; u < 4; u++) {
            int cx = min(max(ix - 1 + u, 0), nx - 1);
            for (int v = 0; v < 4; v++) {
                int cy = min(max(iy - 1 + v, 0), ny - 1);
                for (int w = 0; w < 4; w++) {
                    int cz = min(max(iz - 1 + w, 0), nz - 1);
                    int off = cx * nyz + cy * nz + cz;
                    real va = sliceA[off], vb = sliceB[off];
                    real ww = wx[u] * wy[v] * wz[w];
                    real dxw = dwx[u] * wy[v] * wz[w];
                    real dyw = wx[u] * dwy[v] * wz[w];
                    real dzw = wx[u] * wy[v] * dwz[w];
                    a += ww * va; ax += dxw * va; ay += dyw * va; az += dzw * va;
                    b += ww * vb; bx += dxw * vb; by += dyw * vb; bz += dzw * vb;
                }
            }
        }
        *vA = a; *gAx = ax * invSp; *gAy = ay * invSp; *gAz = az * invSp;
        *vB = b; *gBx = bx * invSp; *gBy = by * invSp; *gBz = bz * invSp;
        return;
    }

    *vA = sampleField(sliceA, nx, ny, nz, ox, oy, oz, spacing, method, px, py, pz, 1,
                      gAx, gAy, gAz);
    *vB = sampleField(sliceB, nx, ny, nz, ox, oy, oz, spacing, method, px, py, pz, 1,
                      gBx, gBy, gBz);
}

// Iterate the 3x3x3 cell neighbourhood around (px,py,pz).
#define FOR_EACH_NEAR_POCKET(px, py, pz, ...)                                  \
    {                                                                           \
        real _inv = (real) 1 / cellSize;                                        \
        int _cx = (int) floor((px - cellOriginX) * _inv);                       \
        int _cy = (int) floor((py - cellOriginY) * _inv);                       \
        int _cz = (int) floor((pz - cellOriginZ) * _inv);                       \
        for (int _ix = _cx - 1; _ix <= _cx + 1; _ix++) {                        \
            if (_ix < 0 || _ix >= cellCountX) continue;                         \
            for (int _iy = _cy - 1; _iy <= _cy + 1; _iy++) {                    \
                if (_iy < 0 || _iy >= cellCountY) continue;                     \
                for (int _iz = _cz - 1; _iz <= _cz + 1; _iz++) {                \
                    if (_iz < 0 || _iz >= cellCountZ) continue;                 \
                    int _cell = (_ix * cellCountY + _iy) * cellCountZ + _iz;    \
                    int _b = cellStart[_cell], _e = cellStart[_cell + 1];       \
                    for (int _p = _b; _p < _e; _p++) {                          \
                        int j = cellAtoms[_p];                                  \
                        __VA_ARGS__                                             \
                    }                                                           \
                }                                                               \
            }                                                                   \
        }                                                                       \
    }

// =============================================================================
// Offline field generation
// =============================================================================

/**
 * Phi_k(x) = sum_j q_j S(|x-r_j|) / f_GB(|x-r_j|, R_k, R_apo_j)
 * One thread per grid point; every receptor atom contributes, because the
 * field decays like 1/r and cannot be truncated at the pocket boundary.
 * Output layout: gridOut[slice * totalGridPoints + gridIdx].
 */
#define MAX_FIELD_SLICES 16
extern "C" __global__ void generateCrossFieldSlices(
    float* __restrict__ gridOut,
    const real4* __restrict__ receptorPositions,
    const real* __restrict__ receptorCharges,
    const real* __restrict__ receptorBornApo,
    int numReceptorAtoms,
    const float* __restrict__ sliceRadii,
    int numSlices,
    float originX, float originY, float originZ,
    int nx, int ny, int nz,
    float spacing,
    float switchOn, float switchOff,
    int totalGridPoints
) {
    for (int gridIdx = blockIdx.x * blockDim.x + threadIdx.x;
         gridIdx < totalGridPoints; gridIdx += gridDim.x * blockDim.x) {

        int nyz = ny * nz;
        int ix = gridIdx / nyz;
        int rem = gridIdx % nyz;
        int iy = rem / nz;
        int iz = rem % nz;
        real px = originX + ix * spacing;
        real py = originY + iy * spacing;
        real pz = originZ + iz * spacing;

        real accum[MAX_FIELD_SLICES];
        for (int k = 0; k < numSlices; k++) accum[k] = (real) 0;

        for (int j = 0; j < numReceptorAtoms; j++) {
            real4 rj = receptorPositions[j];
            real dx = px - rj.x, dy = py - rj.y, dz = pz - rj.z;
            real r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < (real) 1e-20) continue;
            real r = sqrt(r2);
            real sw = fieldSwitch(r, switchOn, switchOff);
            if (sw == (real) 0) continue;
            real qs = receptorCharges[j] * sw;
            real Rj = receptorBornApo[j];
            for (int k = 0; k < numSlices; k++)
                accum[k] += qs * invFgb(r2, (real) sliceRadii[k], Rj);
        }

        for (int k = 0; k < numSlices; k++)
            gridOut[(size_t) k * totalGridPoints + gridIdx] = (float) accum[k];
    }
}

/**
 * Psi_b(x) = sum_j w_j H(|x-r_j|, rho_off_j, s_b) S(|x-r_j|)
 * H decays fast enough to truncate, so only atoms within buildCutoff of the
 * point contribute. Receptor arrays here are the mirror-build selection.
 */
extern "C" __global__ void generateMirrorFieldSlices(
    float* __restrict__ gridOut,
    const real4* __restrict__ atomPositions,
    const real* __restrict__ atomRadii,
    const real* __restrict__ atomWeights,
    int numAtomsIn,
    const float* __restrict__ sliceRadii,
    int numSlices,
    float originX, float originY, float originZ,
    int nx, int ny, int nz,
    float spacing,
    float switchOn, float switchOff, float buildCutoff,
    int totalGridPoints
) {
    for (int gridIdx = blockIdx.x * blockDim.x + threadIdx.x;
         gridIdx < totalGridPoints; gridIdx += gridDim.x * blockDim.x) {

        int nyz = ny * nz;
        int ix = gridIdx / nyz;
        int rem = gridIdx % nyz;
        int iy = rem / nz;
        int iz = rem % nz;
        real px = originX + ix * spacing;
        real py = originY + iy * spacing;
        real pz = originZ + iz * spacing;
        real cutoff2 = buildCutoff * buildCutoff;

        real accum[MAX_FIELD_SLICES];
        for (int k = 0; k < numSlices; k++) accum[k] = (real) 0;

        for (int j = 0; j < numAtomsIn; j++) {
            real w = atomWeights[j];
            if (w == (real) 0) continue;
            real4 rj = atomPositions[j];
            real dx = px - rj.x, dy = py - rj.y, dz = pz - rj.z;
            real r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > cutoff2 || r2 < (real) 1e-20) continue;
            real r = sqrt(r2);
            real sw = fieldSwitch(r, switchOn, switchOff);
            if (sw == (real) 0) continue;
            real Rj_off = atomRadii[j] - DIELECTRIC_OFFSET;
            real ws = w * sw;
            for (int k = 0; k < numSlices; k++)
                accum[k] += ws * hctTerm(r, Rj_off, (real) sliceRadii[k]);
        }

        for (int k = 0; k < numSlices; k++)
            gridOut[(size_t) k * totalGridPoints + gridIdx] = (float) accum[k];
    }
}

// =============================================================================
// Runtime: cross term (CROSS_RADIUS_GRID)
// =============================================================================

/**
 * Pass A: accumulate the ligand-induced HCT on near-shell pocket atoms.
 * One thread per ligand atom; scatters into recDeltaHCT[group * numPocket + j].
 */
extern "C" __global__ void accumulateReceptorNearHCT(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ ligandRadii,
    const real* __restrict__ ligandScaleFactors,
    const int* __restrict__ groupStart,
    int numGroups, int templateNumAtoms, int totalLigandAtoms,
    const real4* __restrict__ pocketPositions,
    const real* __restrict__ pocketRadii,
    int numPocket,
    const int* __restrict__ cellStart,
    const int* __restrict__ cellAtoms,
    float cellOriginX, float cellOriginY, float cellOriginZ,
    float cellSize, int cellCountX, int cellCountY, int cellCountZ,
    float nearCutoff,
    real* __restrict__ recDeltaHCT
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalLigandAtoms) return;
    int atomInGroup;
    int groupIdx = findGroup(numGroups, templateNumAtoms, idx, &atomInGroup);
    if (groupIdx < 0) return;

    int tmpl = atomInGroup % templateNumAtoms;
    real4 p = posq[particleIndices[idx]];
    real s_i = (ligandRadii[tmpl] - DIELECTRIC_OFFSET) * ligandScaleFactors[tmpl];
    real cutoff2 = nearCutoff * nearCutoff;
    real* out = recDeltaHCT + (size_t) groupIdx * numPocket;

    FOR_EACH_NEAR_POCKET(p.x, p.y, p.z, {
        real4 rj = pocketPositions[j];
        real dx = p.x - rj.x, dy = p.y - rj.y, dz = p.z - rj.z;
        real r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > cutoff2 || r2 < (real) 1e-20) continue;
        real r = sqrt(r2);
        real Rj_off = pocketRadii[j] - DIELECTRIC_OFFSET;
        atomicAdd(&out[j], hctTerm(r, Rj_off, s_i));
    })
}

/**
 * Pass B: cross energy, explicit forces, dE/dR_i, and the accumulation of
 * dE/dR'_j that pass C turns into forces.
 */
extern "C" __global__ void computeCrossRadiusGridEnergy(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ charges,
    const real* __restrict__ bornRadii,
    const int* __restrict__ groupStart,
    int numGroups, int templateNumAtoms, int totalLigandAtoms,
    real prefactor,
    const float* __restrict__ crossField,
    const float* __restrict__ sliceRadii,
    int numSlices, int totalGridPoints,
    float originX, float originY, float originZ,
    int nx, int ny, int nz, float spacing, int interpMethod,
    const real4* __restrict__ pocketPositions,
    const real* __restrict__ pocketCharges,
    const real* __restrict__ pocketRadii,
    const real* __restrict__ pocketApoHCT,
    const real* __restrict__ pocketBornApo,
    int numPocket,
    const int* __restrict__ cellStart,
    const int* __restrict__ cellAtoms,
    float cellOriginX, float cellOriginY, float cellOriginZ,
    float cellSize, int cellCountX, int cellCountY, int cellCountZ,
    float nearCutoff, float switchOn, float switchOff, int useOBC,
    const real* __restrict__ recDeltaHCT,
    real* __restrict__ dCrossDRrec,
    real* __restrict__ dE_dR,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms,
    mixed* __restrict__ groupCrossEnergies,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalLigandAtoms) return;
    int atomInGroup;
    int groupIdx = findGroup(numGroups, templateNumAtoms, idx, &atomInGroup);
    if (groupIdx < 0) return;

    int tmpl = atomInGroup % templateNumAtoms;
    int particleIdx = particleIndices[idx];
    real4 p = posq[particleIdx];
    real qi = charges[tmpl];
    real Ri = bornRadii[idx];
    real scale = globalScalingFactor * groupScalingFactors[groupIdx];
    const real* dI = recDeltaHCT + (size_t) groupIdx * numPocket;
    real* dRrec = dCrossDRrec + (size_t) groupIdx * numPocket;

    // Bracket R_i between two slices, linear in R.
    real Rlo = (real) sliceRadii[0], Rhi = (real) sliceRadii[numSlices - 1];
    real Rc = min(max(Ri, Rlo), Rhi);
    int k = numSlices - 2;
    for (int t = 0; t + 1 < numSlices; t++) {
        if (Rc <= (real) sliceRadii[t + 1]) { k = t; break; }
    }
    real Rk = (real) sliceRadii[k], Rk1 = (real) sliceRadii[k + 1];
    real invDR = (real) 1 / (Rk1 - Rk);
    real wHi = (Rc - Rk) * invDR;

    real gxL, gyL, gzL, gxH, gyH, gzH, phiLo, phiHi;
    sampleFieldPair(crossField + (size_t) k * totalGridPoints,
                    crossField + (size_t) (k + 1) * totalGridPoints,
                    nx, ny, nz, originX, originY, originZ, spacing,
                    interpMethod, p.x, p.y, p.z,
                    &phiLo, &gxL, &gyL, &gzL, &phiHi, &gxH, &gyH, &gzH);

    real energy = prefactor * qi * ((real) 1 - wHi) * phiLo + prefactor * qi * wHi * phiHi;
    real fx = -prefactor * qi * (((real) 1 - wHi) * gxL + wHi * gxH);
    real fy = -prefactor * qi * (((real) 1 - wHi) * gyL + wHi * gyH);
    real fz = -prefactor * qi * (((real) 1 - wHi) * gzL + wHi * gzH);

    bool inBracket = (Ri > Rlo && Ri < Rhi);
    real dEdRi = inBracket ? (prefactor * qi * (phiHi - phiLo) * invDR) : (real) 0;

    real cutoff2 = nearCutoff * nearCutoff;
    FOR_EACH_NEAR_POCKET(p.x, p.y, p.z, {
        real4 rj = pocketPositions[j];
        real dx = p.x - rj.x, dy = p.y - rj.y, dz = p.z - rj.z;
        real r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > cutoff2 || r2 < MIN_CROSS_R2) continue;
        real r = sqrt(r2);

        real Ra = pocketBornApo[j];
        real Rj = Ra;
        if (dI[j] != (real) 0)
            Rj = bornFromHCT(pocketRadii[j], pocketApoHCT[j] + dI[j], useOBC);

        real D = Ri * Rj;
        real alpha = r2 / ((real) 4 * D);
        real expA = exp(-alpha);
        real fgb2 = r2 + D * expA;
        real fgb = sqrt(fgb2);
        real qq = prefactor * qi * pocketCharges[j];

        real sw = fieldSwitch(r, switchOn, switchOff);
        real farLo = (real) 0, farHi = (real) 0, farTerm = (real) 0;
        if (sw != (real) 0) {
            farLo = invFgb(r2, Rk, Ra);
            farHi = invFgb(r2, Rk1, Ra);
            farTerm = sw * (((real) 1 - wHi) * farLo + wHi * farHi);
        }
        energy += qq * ((real) 1 / fgb - farTerm);

        real dfgb_dr = r * ((real) 4 - expA) / ((real) 4 * fgb);
        real dTerm_dr = -dfgb_dr / fgb2;
        if (sw != (real) 0) {
            real dsw = fieldSwitchDeriv(r, switchOn, switchOff);
            real fLo = (real) 1 / farLo, fHi = (real) 1 / farHi;
            real aLo = r2 / ((real) 4 * Rk * Ra), aHi = r2 / ((real) 4 * Rk1 * Ra);
            real dLo = -r * ((real) 4 - exp(-aLo)) / ((real) 4 * fLo * fLo * fLo);
            real dHi = -r * ((real) 4 - exp(-aHi)) / ((real) 4 * fHi * fHi * fHi);
            real farVal = ((real) 1 - wHi) * farLo + wHi * farHi;
            dTerm_dr -= dsw * farVal + sw * (((real) 1 - wHi) * dLo + wHi * dHi);
        }
        real dE_dr = qq * dTerm_dr;
        real invR = (real) 1 / r;
        fx -= dE_dr * dx * invR;
        fy -= dE_dr * dy * invR;
        fz -= dE_dr * dz * invR;

        real dfgb_dRi = Rj * expA * ((real) 1 + alpha) / ((real) 2 * fgb);
        dEdRi += -qq / fgb2 * dfgb_dRi;
        if (sw != (real) 0 && inBracket)
            dEdRi -= qq * sw * (farHi - farLo) * invDR;

        if (dI[j] != (real) 0) {
            real dfgb_dRj = Ri * expA * ((real) 1 + alpha) / ((real) 2 * fgb);
            atomicAdd(&dRrec[j], -qq / fgb2 * dfgb_dRj);
        }
    })

    atomicAdd(&forceBuffer[particleIdx],
              (unsigned long long) ((long long) (fx * scale * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + paddedNumAtoms],
              (unsigned long long) ((long long) (fy * scale * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + 2 * paddedNumAtoms],
              (unsigned long long) ((long long) (fz * scale * 0x100000000)));
    dE_dR[idx] += dEdRi;
    if (groupCrossEnergies != 0)
        atomicAdd(&groupCrossEnergies[groupIdx], (mixed) (energy * scale));
}

/**
 * Pass C: forces from the ligand's own perturbation of the receptor radii,
 * dE/dR'_j * dR'_j/dhct_j * dhct_j/dx_i. The receptor is rigid, so only the
 * ligand feels it.
 */
extern "C" __global__ void applyCrossReceptorChainRule(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ ligandRadii,
    const real* __restrict__ ligandScaleFactors,
    const int* __restrict__ groupStart,
    int numGroups, int templateNumAtoms, int totalLigandAtoms,
    const real4* __restrict__ pocketPositions,
    const real* __restrict__ pocketRadii,
    const real* __restrict__ pocketApoHCT,
    int numPocket,
    const int* __restrict__ cellStart,
    const int* __restrict__ cellAtoms,
    float cellOriginX, float cellOriginY, float cellOriginZ,
    float cellSize, int cellCountX, int cellCountY, int cellCountZ,
    float nearCutoff, int useOBC,
    const real* __restrict__ recDeltaHCT,
    const real* __restrict__ dCrossDRrec,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalLigandAtoms) return;
    int atomInGroup;
    int groupIdx = findGroup(numGroups, templateNumAtoms, idx, &atomInGroup);
    if (groupIdx < 0) return;

    int tmpl = atomInGroup % templateNumAtoms;
    int particleIdx = particleIndices[idx];
    real4 p = posq[particleIdx];
    real s_i = (ligandRadii[tmpl] - DIELECTRIC_OFFSET) * ligandScaleFactors[tmpl];
    real scale = globalScalingFactor * groupScalingFactors[groupIdx];
    const real* dI = recDeltaHCT + (size_t) groupIdx * numPocket;
    const real* dRrec = dCrossDRrec + (size_t) groupIdx * numPocket;
    real cutoff2 = nearCutoff * nearCutoff;

    real fx = 0, fy = 0, fz = 0;
    FOR_EACH_NEAR_POCKET(p.x, p.y, p.z, {
        if (dI[j] == (real) 0 || dRrec[j] == (real) 0) continue;
        real4 rj = pocketPositions[j];
        real dx = p.x - rj.x, dy = p.y - rj.y, dz = p.z - rj.z;
        real r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > cutoff2 || r2 < (real) 1e-20) continue;
        real r = sqrt(r2);
        real rho = pocketRadii[j];
        real total = pocketApoHCT[j] + dI[j];
        real born = bornFromHCT(rho, total, useOBC);
        real factor = dRrec[j] * dBornDHCT(rho, total, born, useOBC);
        if (factor == (real) 0) continue;
        real dHCT = hctTermDeriv(r, rho - DIELECTRIC_OFFSET, s_i);
        real mag = factor * dHCT / r;
        fx -= mag * dx;
        fy -= mag * dy;
        fz -= mag * dz;
    })

    atomicAdd(&forceBuffer[particleIdx],
              (unsigned long long) ((long long) (fx * scale * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + paddedNumAtoms],
              (unsigned long long) ((long long) (fy * scale * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + 2 * paddedNumAtoms],
              (unsigned long long) ((long long) (fz * scale * 0x100000000)));
}

// =============================================================================
// Runtime: mirror term (MIRROR_LINEAR_GRID)
// =============================================================================

/**
 * Field lookup on the atom's descreener slice plus the near-shell remainder
 * the switch removed. Independent of the ligand Born radii, so it feeds no
 * chain rule.
 */
extern "C" __global__ void computeMirrorFromField(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const int* __restrict__ atomMirrorSlice,
    const int* __restrict__ groupStart,
    int numGroups, int templateNumAtoms, int totalLigandAtoms,
    const float* __restrict__ mirrorField,
    const float* __restrict__ sliceRadii,
    int totalGridPoints,
    float originX, float originY, float originZ,
    int nx, int ny, int nz, float spacing, int interpMethod,
    const real4* __restrict__ pocketPositions,
    const real* __restrict__ pocketRadii,
    const real* __restrict__ pocketWeights,
    int numPocket,
    const int* __restrict__ cellStart,
    const int* __restrict__ cellAtoms,
    float cellOriginX, float cellOriginY, float cellOriginZ,
    float cellSize, int cellCountX, int cellCountY, int cellCountZ,
    float switchOn, float switchOff, float mirrorScale,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms,
    mixed* __restrict__ groupMirrorEnergies,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalLigandAtoms) return;
    int atomInGroup;
    int groupIdx = findGroup(numGroups, templateNumAtoms, idx, &atomInGroup);
    if (groupIdx < 0) return;

    int tmpl = atomInGroup % templateNumAtoms;
    int particleIdx = particleIndices[idx];
    real4 p = posq[particleIdx];
    int slice = atomMirrorSlice[tmpl];
    real s_i = (real) sliceRadii[slice];
    real scale = globalScalingFactor * groupScalingFactors[groupIdx];

    real gx, gy, gz;
    real energy = sampleField(mirrorField + (size_t) slice * totalGridPoints,
                              nx, ny, nz, originX, originY, originZ, spacing,
                              interpMethod, p.x, p.y, p.z, 1, &gx, &gy, &gz);
    real fx = -gx, fy = -gy, fz = -gz;

    real cutoff2 = switchOff * switchOff;
    FOR_EACH_NEAR_POCKET(p.x, p.y, p.z, {
        real w = pocketWeights[j];
        if (w == (real) 0) continue;
        real4 rj = pocketPositions[j];
        real dx = p.x - rj.x, dy = p.y - rj.y, dz = p.z - rj.z;
        real r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > cutoff2 || r2 < (real) 1e-20) continue;
        real r = sqrt(r2);
        real Rj_off = pocketRadii[j] - DIELECTRIC_OFFSET;
        real sw = fieldSwitch(r, switchOn, switchOff);
        real H = hctTerm(r, Rj_off, s_i);
        energy += w * H * ((real) 1 - sw);

        real dH = hctTermDeriv(r, Rj_off, s_i);
        real dsw = fieldSwitchDeriv(r, switchOn, switchOff);
        real dE_dr = w * (dH * ((real) 1 - sw) - H * dsw);
        real invR = (real) 1 / r;
        fx -= dE_dr * dx * invR;
        fy -= dE_dr * dy * invR;
        fz -= dE_dr * dz * invR;
    })

    real s = scale * mirrorScale;
    atomicAdd(&forceBuffer[particleIdx],
              (unsigned long long) ((long long) (fx * s * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + paddedNumAtoms],
              (unsigned long long) ((long long) (fy * s * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + 2 * paddedNumAtoms],
              (unsigned long long) ((long long) (fz * s * 0x100000000)));
    if (groupMirrorEnergies != 0)
        atomicAdd(&groupMirrorEnergies[groupIdx], (mixed) (energy * s));
}
