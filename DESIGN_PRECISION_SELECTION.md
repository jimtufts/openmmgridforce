# Precision Selection Refactor — Scoping Document

Goal: make the CUDA kernels honor OpenMM's precision-mode conventions
(`single` / `mixed` / `double`) instead of being hardwired to fp32. This is a
scoping/analysis document, not an implementation plan to execute yet.

---

## Part A — How OpenMM handles precision (reference)

OpenMM's CUDA platform has **three precision modes**, selected by a single
`Platform` property `"Precision"` (legacy alias `"CudaPrecision"`), **default
`"single"`**:

| Aspect                         | single | mixed              | double |
|--------------------------------|:------:|:------------------:|:------:|
| Force/energy *math* (`real`)   | float  | **float**          | double |
| Integration / accum (`mixed`)  | float  | **double**         | double |
| Position storage (`posq`)      | float4 | float4 + `posqCorrection` (float4) | double4 |
| Velocity/inv-mass (`velm`)     | float4 | **double4**        | double4 |
| Energy accumulation buffer     | float  | **double**         | double |
| Force accumulator buffer       | `long long` fixed-point | `long long` fixed-point | `long long` fixed-point |

Two booleans drive everything (`CudaContext`): `useDoublePrecision`,
`useMixedPrecision`. From them OpenMM derives, **and injects into every kernel
compiled through `CudaContext::createModule`**:

- **Typedefs:** `real`/`real2`/`real3`/`real4` (= `double*` only in double mode;
  `float*` in single *and* mixed) and `mixed`/`mixed2/3/4` (= `double*` in mixed
  *and* double; `float*` only in single).
- **Constructor macros:** `make_real3` → `make_float3`/`make_double3`,
  `make_mixed3` likewise.
- **Math intrinsics:** `SQRT`/`RSQRT`/`RECIP`/`EXP`/`LOG`/`POW`/`SIN`/`COS`/
  `ERF`/`FMA`/`FABS` → single- or double-precision variant.
- **Feature macros:** `USE_DOUBLE_PRECISION=1` (double only),
  `USE_MIXED_PRECISION=1` (mixed only); neither in single mode.
- Plus `vectorOps.cu` operator/`dot`/`cross`/`normalize`/`trimTo3/4` helpers.

**Rule of thumb for writing kernels the OpenMM way:**
- `real` for bulk per-particle data (positions, params, interpolation math).
- `mixed` for *accumulation* that benefits from extra precision in mixed mode.
- Forces → `long long` fixed-point: `realToFixedPoint(x) = (long long)(x * 0x100000000)`
  (scale 2³²), 64-bit `atomicAdd`, convert back with `1/(real)0x100000000`.
  This is what buys deterministic, order-independent force summation.
- Mixed mode positions = compensated double-float: base `posq` (float4) +
  residual `posqCorrection` (float4), reconstructed as `posq + posqCorrection`
  in a `mixed`. Guard with `#ifdef USE_MIXED_PRECISION`.

Sources: `openmm/openmm` master — `platforms/cuda/src/CudaContext.cpp`
(parsing ~92-105, typedef emission ~475-515, intrinsics ~279-294),
`CudaPlatform.cpp` (default `single` L129), `kernels/common.cu`
(`realToFixedPoint`), `kernels/vectorOps.cu`, `common/src/kernels/
integrationUtilities.cc` (`loadPos`/`storePos`/`loadForce`). User guide:
docs.openmm.org/latest/userguide/library/04_platform_specifics.html.

---

## Part B — Current state of this project

### The good news: the infrastructure is already free

This plugin runs **inside** OpenMM's CUDA platform. `cu` is OpenMM's
`CudaContext` (`CudaGridForceKernelFactory.cpp:51`), and every kernel here is
compiled with `cu.createModule(...)`. That means **`real`, `mixed`,
`make_real3`, `SQRT`, `USE_MIXED_PRECISION`, etc. are already injected into our
kernel sources** — we just aren't using them in most kernels. We do **not** need
to build a precision-selection mechanism, parse a property, or plumb `-D` flags.
The mechanism already exists; we are adopting it.

Proof it's already partly used: `CudaMultiGroupHMCKernels.cpp:141` and the
NUTS kernels already branch on `cu.getUseDoublePrecision() ||
cu.getUseMixedPrecision()`.

### The current precision picture, by kernel group

| Group | Files | Current types | State |
|---|---|---|---|
| **Grid force/hessian** | `gridForce.cu`, `gridForceTiled.cu`, `gridGeneration.cu`, `gridHessian*.cu` | hardcoded `float4`/`float3`/`float*` | **fp32-locked; breaks in double** |
| **GBSA grid** | `gbsaGridGeneration.cu`, `gbsaGridForce.cu`, `gbsaHessianDouble.cu` | `float` params, some explicit `double` intermediates | fp32-locked + ad-hoc double |
| **Isolated** | `isolatedNonbonded.cu`, `isolatedBonded.cu`, `isolatedSite.cu`, `isolatedGBSA.cu`, `bondedHessian.cu` | `real4`/`real` for params, `float` energies | partially abstracted, inconsistent |
| **HMC/NUTS** | `multiGroupHMC.cu`, `multiGroupNUTS.cu`, `metricAssembly.cu` | branch on precision already | closest to convention |
| **Shared headers** | `include/*.cuh` (interpolation, chain rules, coeffs) | `float` throughout | fp32-locked; touched by many kernels |

### Fixed-point scales currently in use
- Forces & energies: `0x100000000` (2³²) — matches OpenMM. ✅
- Hessian: `0x1000000` (2²⁴) — custom, for ~10⁶ kJ/mol/nm² magnitudes.
- Unscale host-side: `1.0 / 0x100000000` (`metricAssembly.cu:357`).

### ⚠️ Latent correctness bug (independent of this refactor)
`gridForce.cu:17` declares `const float4* __restrict__ posq`, but
`cu.getPosq()` is a **`double4` buffer when the OpenMM context is `double`
precision**. Any user running their `Context` in double today would have the
grid/GBSA/isolated-grid kernels silently reinterpret `double4` as `float4` →
garbage positions, no error. Same risk wherever a kernel hardcodes `float4` for
`posq`/`velm`. Today these kernels are only correct in `single` and `mixed`
(both store `posq` as float4). **This refactor is also the fix for that bug.**

---

## Part B.5 — Precision policy (the design we're adopting)

This project does **not** want a single global precision mode. It wants a small
**precision policy** of independent axes, each defaulting to "follow the OpenMM
context mode" but individually overridable. Rationale: in a grid-based method the
*stored field* is the dominant accuracy lever and the *Hessian → eigendecomp →
entropy* path is the most error-sensitive — neither should be hostage to the
force-math precision, and neither should force the user into full `double` mode
(which slows everything) to get accuracy where it actually matters.

| Axis | Governs | Compile knob (proposed) | Default | Why decoupled |
|---|---|---|---|---|
| **Compute** | force/energy math + general accumulation | OpenMM `real`/`mixed` (injected by `cu.createModule`) | OpenMM context `Precision` | this is OpenMM's own knob — consume, don't reinvent |
| **Grid storage** | f32/f64 of the field array on device | `GRID_STORAGE_TYPE` | follow `real` | grid is static input; fidelity is the top accuracy lever, orthogonal to per-step compute cost |
| **Grid interp accum** | summation type during interpolation | `mixed` | follow `mixed` | f64 accumulation in mixed mode recovers most of a double grid's benefit cheaply |
| **Hessian compute** | 2nd-derivative math | `HESSIAN_TYPE` | f64 | feeds eigendecomp/entropy; double already needed empirically (`gbsaHessianDouble.cu`) |
| **Hessian accum** | fixed-point scale or native double | `HESSIAN_SCALE` / native | adjustable | 2²⁴ fixed-point floor (~6e-8) corrupts soft-mode eigenvalues |

Key freedom: the **grid storage** and **Hessian** axes are genuinely independent
of compute precision. The most useful production combo may be `mixed` compute +
**f64 grid** + **f64 Hessian** — fast float force math, high-fidelity field, and
trustworthy entropy, without paying for double everywhere.

Determinism note (distinguish *determinism* = same answer every run, from
*accuracy* = close to truth — the Hessian needs both):
- **Forces** stay fixed-point: HMC/NUTS reversibility needs order-independent
  sums. Non-negotiable.
- **Hessian stays deterministic by default**, driven by **normal-mode / entropy
  reproducibility**, not the sampler. Soft (low-frequency) modes are where
  accumulation-order noise shows up (large λ swamp it, near-zero λ don't;
  frequency ∝ √λ), and vibrational entropy is dominated by those soft modes — so
  float-atomic non-determinism produces run-to-run entropy jitter and breaks
  regression tests / result caching. A tool that returns a different entropy each
  run on identical input is unacceptable.
- The RMHMC metric (`metricAssembly.cu:2-6,72-95`) is a *second* consumer that
  would need determinism for accept/reject correctness — but it's an experimental
  path, so it's handled by a **guard that errors if the metric is assembled from
  a non-deterministic Hessian**, not by constraining the default.
- Consequence: native non-deterministic double atomics are **off the table** for
  the Hessian default. Fix the precision floor *inside* the deterministic scheme
  (Tier 4). A non-deterministic high-precision mode, if ever built, is an
  explicit opt-in that the metric guard rejects.

How the knobs reach the kernels: they become plugin-level settings (on the Force
objects or context-creation params) that populate the `defines` map already
passed to every `cu.createModule(...)` call — independent of OpenMM's injected
`real`/`mixed`. Host API stays `double` in/out regardless.

---

## Part C — Scope of work

### Tier 1 — Position/velocity boundary (correctness-critical, small)
Make every kernel read `posq`/`velm` at the context's element size.
- Change `float4* posq` → `real4* posq` (or `mixed4` reconstruction where the
  HMC/NUTS path needs full position precision in mixed mode via
  `posqCorrection`).
- Audit every `cu.getPosq()` / `cu.getVelm()` consumer (`grep` shows ~20 call
  sites across grid, GBSA, isolated, HMC, NUTS).
- This alone fixes the latent double-precision bug.

### Tier 2 — Grid kernels float → real/mixed (the bulk)
- `gridForce.cu`, `gridForceTiled.cu`, `gridHessian*.cu`, `gridGeneration.cu`:
  convert interpolation math to `real`, accumulation to `mixed`, intrinsics to
  `SQRT`/`EXP`/etc.
- Shared headers `include/GridInterpolation.cuh`, `InterpolationBasis.cuh`,
  `Hermite*`, `Tricubic*`, `Triquintic*`, `InvPowerChainRule.cuh`,
  `*ChainRule.cuh`: these are included by many kernels, so converting them is
  high-leverage but also the highest-blast-radius change. **Decide whether to
  templatize on a type or rely on the injected `real`/`mixed` typedefs**
  (the latter matches OpenMM and is simpler — no template instantiation).

### Tier 3 — Decoupled grid *storage* precision (`GRID_STORAGE_TYPE`)
Grid values arrive from the C++/Python API as `double` and are currently
converted to a `float*` device buffer. Make grid storage a **decoupled knob**
(`GRID_STORAGE_TYPE` ∈ {f32, f64}, default = follow `real`), independent of
compute precision. Interpolation **accumulates in `mixed`** regardless, so an
f64 grid is double-accurate even under float (`mixed`-mode) compute.
- Grid device buffers become **element-size-aware** (allocate/upload f32 vs f64
  off `GRID_STORAGE_TYPE`); host API stays `double`, downcast at upload only when
  storage is f32.
- Kernel load path reads `GRID_STORAGE_TYPE`, casts node values to `mixed` for
  the interpolation sum.
- **Tiled-streaming cache** (`TileManager`/`TileCache`) budget math must use the
  actual grid element size, not a hardcoded 4 bytes — f64 storage halves tiles
  per VRAM budget. **TODO: verify current sizing isn't hardcoded to 4.**
- Grid-generation kernels write `GRID_STORAGE_TYPE`.
- Why decoupled (not welded to `real`): the stored field is the dominant accuracy
  lever; this lets `mixed` compute + f64 grid (fast math, high-fidelity field)
  without forcing full `double` mode. Cost is one define + a load-time cast.
- **Scope extension — the axis must also cover the interpolation COEFFICIENTS and
  eval ACCUMULATION, not just stored grid values.** See the motivating case below.

#### Motivating case: triquintic-Hermite minimizer failure (`precision_issues.md`)
Switching receptor-grid interpolation from cubic B-spline to triquintic Hermite
breaks production: at 0.25 Å, ~25-30% of ligands NaN and survivors hang in
`LocalEnergyMinimizer` for >1 h; at 0.15 Å, only **1.3%** of sampled wells are
positive-definite (B-spline: 100%). The generation path is analytical-correct and
the runtime kernel computes E and F from the same coefficients, so it is a
**precision/self-consistency** failure, not an accuracy one:
- The 216 per-cell quintic coefficients come from a 216×216 solve
  (`TriquinticCoefficients.cuh`) over the 27 stored corner derivatives, all in
  fp32. fp32 roundoff (~1e-5) means adjacent cells' polynomials don't *exactly*
  agree on their shared face → tiny E/F discontinuities at every cell boundary.
- L-BFGS line search needs ~1e-6 consistency; ~1e-5 stalls it → it bails at
  non-minima → non-PD "fake" Hessians and hangs.
- ⇒ The grid-precision knob must apply to **coefficient assembly (f64 216×216
  solve)** and **eval accumulation (`mixed`)**, since that is where the E/F
  inconsistency is born — storing f64 grid *values* alone would not fix it.
- ⚠️ The bug is NOT yet fully root-caused. The FD consistency test in that doc is
  **not** valid evidence (central-difference fp32 cancellation blows up as h→0
  regardless of kernel correctness). Localize with **JAX autodiff** vs the
  plugin's analytical F/Hessian (1D slice first, then 3D).
- **1D JAX result (`python/tests/test_triquintic_hermite_1d_jax.py`):**
  - Polynomial math is **correct** — quintic Hermite reproduces E/E'/E'' at nodes
    (~1e-12), jax.grad matches the hand-coded analytical force (~1e-14), and it is
    **C2 across cells in f64** (force jump ~1e-13). So this is *not* an algorithmic
    bug, at least in 1D.
  - **fp32 node storage alone breaks it:** casting node data to float32 reintroduces
    a cross-cell **force discontinuity of ~1.5e-6 to 2e-5**, exceeding the ~1e-6
    L-BFGS line search needs — reproducing the production failure mechanism. And
    this is *before* the 3D 216×216 coefficient solve, which can only add error.
  - Mechanism note: the jump scales ~1/h, so it is **worse at finer spacing** —
    consistent with the production failures at 0.15/0.25 Å. B-spline avoids it
    because its prefiltered control points are shared globally (one C2 surface), vs.
    Hermite reconstructing each cell independently from stored derivatives.
  - ⇒ Strong support that **f64 grid/derivative storage is the fix.**
- **3D JAX result (`python/tests/test_triquintic_hermite_3d_jax.py`,** real 1g9v
  charge grid, stored derivatives read via `getDerivatives()`, layout deriv-major):
  - **Kernel algorithm is correct** — plugin energy reconstructs the JAX (f64)
    polynomial from the same stored corners to **rel 5.9e-7**. So the
    `TriquinticCoefficients` matrix, derivative-major X layout, monomial indexing
    and eval are all right; this is **not** an algorithmic/matrix bug. (Plugin
    force matches to ~4e-4, dominated by fp32 force-eval + the plugin's f32
    sub-cell position — a precision diagnostic, not a bug.)
  - **The fp32 coefficient solve is the culprit.** Assembling `a = 0.125·M·X` in
    f64 gives a true-C2 shared face (force jump **2e-13**); assembling it in
    **fp32** (the kernel's actual path — `M` is `const float`, `X`/`a` are float)
    gives a cross-cell **force jump ≈ 6 kJ/mol/nm** (value jump ~6e-2) — **6+
    orders above the ~1e-6 L-BFGS line search needs.** This reproduces and
    explains the production hangs / non-PD "fake" Hessians.
  - Scales with |V| (the 216-term f32 matvec over large potential values),
    so it is worst exactly where the doc reports failures: capped close-contact
    regions and finer spacing.
  - ⇒ **The grid-precision axis must cover the coefficient assembly (f64/`mixed`
    216-solve), not just stored values** — confirmed empirically (6 kJ/mol/nm →
    2e-13). Caveat: the exact f32 jump magnitude is model-dependent (numpy f32
    matmul accumulation order vs the kernel's sequential f32 loop); the *order*
    (≫1e-6) and the f64 fix are robust.
- **PROTOTYPE IMPLEMENTED + VALIDATED (in-kernel f64 assembly).** Converted the
  triquintic coefficient assembly + polynomial eval to `double` (keeping fp32
  storage) in `GridInterpolation.cuh::triquinticInterpolate` (the `invPowerMode==0`
  fast path — the one actually used), plus the inv-power inline block in
  `gridForce.cu` and the tiled `triquinticInterpolateTiled` in `gridForceTiled.cu`.
  Measured on the real plugin (`test_triquintic_hermite_3d_jax.py`, section D —
  direct plugin straddle):
  - cross-cell force discontinuity **3.515 kJ/mol/nm → 1.6e-4** (~22,000×, now
    below the straddle estimator's smooth-trend noise floor — i.e. gone).
  - plugin force vs f64 reference **4.4e-4 → 6.1e-7**; energy **5.9e-7 → 1.8e-8**.
  - ⇒ f64 *assembly alone*, with fp32-stored derivatives, eliminates the
    L-BFGS-breaking discontinuity. The residual 1.6e-4 is fp32 storage + fp32
    position/output truncation (the next axes), not assembly.
  - Gotcha: triquintic has assembly sites in the shared `GridInterpolation.cuh`
    (NONE-mode common path), `gridForce.cu` inline (inv-power), `gridForceTiled.cu`
    (tiled), and the two Hessian kernels. OpenMM caches compiled kernels in
    `/tmp/<sha1>_<arch>_64`; clear them when validating kernel edits.
- **HESSIAN PROTOTYPE + VALIDATED.** `gridHessian.cu`/`gridHessianTiled.cu` have
  their OWN inline fp32 216-solve (separate from the force path). Verified the
  corruption is real (`test_triquintic_hessian_precision.py`): the f32 Hessian has
  a **cross-cell discontinuity of 166 kJ/mol/nm²** (vs the force's 3.5 — 2nd
  derivatives amplify the fp32 coeff error) and median per-point eigenvalue error
  6.45. Converting the 216-assembly + eval temporaries to double (accumulators kept
  float for the downstream chain-rule helpers — consistency preserved since both
  cells run identical double→float ops):
  - cross-cell Hessian discontinuity **166 → 0.003 kJ/mol/nm²** (~54,000×).
  - median eigenvalue error **6.45 → 0.017** (~385×).
  - This is the root-cause fix for the non-PD "fake Hessian" symptom (the
    discontinuity/error dwarf the O(0.1–40) soft eigenvalues that decide PD-ness;
    eliminating them removes the spurious negatives).
  - Residual: a few high-curvature points (near receptor atoms, |H|~1e5–1e6) keep
    larger absolute error — that's fp32 *storage* of the derivatives (a separate
    axis), not assembly, and not the C2-continuity issue.
  - Build gotcha: the downstream `applyTanhChainRule`/inv-power helpers take
    `float*`, so the Hessian accumulators must stay `float` (only the assembly +
    eval temporaries go double) or NVRTC fails to compile.
- **END-TO-END FULL-SYSTEM A/B (`repro_fake_pd_fullsystem.py`).** Real ligand
  (1g9v, 47 atoms) minimized on triquintic charge/ljr/lja grids, then full
  Cartesian Hessian (bonded + isolated-NB + grid) eigendecomposed. Same code,
  f32 vs f64 kernels:

  | | f32 (before) | f64 (after) |
  |---|---|---|
  | min energy | 2225 → **173** (stalled) | 2225 → **−575** |
  | max\|force\| after min | **430** (no converge) | **8.9** |
  | Hessian min eigenvalue | **−339** | **+53** |
  | imaginary modes | **3 (NON-PD)** | **0 (PD)** |

  Reproduces BOTH production symptoms in f32 (minimizer stalls at a fake shallow
  point; Hessian non-PD with imaginary modes) and fixes BOTH in f64 (converges to
  a deeper real minimum; Hessian positive definite). This is the literal
  non-PD→PD confirmation; the surgical f64-assembly change is the complete fix for
  the minimizer/fake-PD issues.

### Tier 4 — Hessian precision (decoupled: `HESSIAN_TYPE` + accumulation)
The Hessian feeds eigendecomposition → frequencies/entropy, the most
precision-sensitive path here. Make it a **decoupled axis**, independent of force
precision:
- `HESSIAN_TYPE` ∈ {f32, f64}, default **f64** — generalizes the existing
  hardcoded `gbsaHessianDouble.cu` into policy. Existing explicit-`double`
  intermediates become `HESSIAN_TYPE`.
- **Accumulation stays deterministic fixed-point** (see Part B.5 — normal-mode
  reproducibility requires it; native float/double atomics are out).
- **MEASURED (probe `python/tests/probe_hessian_magnitude.py`, 25 Astex
  ligands, minimized):**
  - max |Hessian entry| ≈ **9.8e5 (~2²⁰)**, bonded-dominated (bond 2k terms);
    nonbonded ≤ ~6e4.
  - softest full-matrix eigenvalue (entropy-relevant) ≈ **0.13 (~2⁻³)**; softest
    per-atom 3×3 block eig ≈ 36.
  - ⇒ A **single 64-bit accumulator comfortably suffices.** Overflow bound
    s ≤ 35; resolution bound (20 rel. bits on softest mode) s ≥ 23.
  - The earlier "2⁻²⁴ floor corrupts soft modes" worry is **overstated**: at
    s=24 the quantum (6e-8) sits ~2²¹ below the softest real mode (0.13). The
    current scale already works.
- **Recommendation:** keep deterministic fixed-point; **bump scale 2²⁴ → 2²⁹–2³⁰**
  for free extra soft-mode margin (overflow still safe to ~2³⁵). **No need** for a
  128-bit/two-word or block-relative accumulator. This de-risks Tier 4 to a
  one-line constant change.
- Caveat: measured at (loosely) minimized poses; genuinely near-saddle poses can
  push an eigenvalue toward 0, but that's a conditioning/physics limit no finite
  scale fixes. Re-run the probe with `--n` larger / on production poses before
  committing the constant.
- **Metric guard:** `assembleMetricTensor` (and the RMHMC/NUTS path) must error
  out if invoked when the Hessian was produced by any non-deterministic mode.
- Verify HCT integrals (GBSA) don't *require* double even when forces are single.

### Tier 5 — GBSA force kernels (cleanup)
Reconcile `gbsaGridForce.cu` / `gbsaGridGeneration.cu` with the `real`/`mixed`
scheme and `GRID_STORAGE_TYPE`. (Hessian intermediates handled in Tier 4.)

### Tier 6 — Isolated kernels (cleanup)
Already use `real` for params; make energies `mixed`, intrinsics consistent,
remove stray `float`. Lowest risk.

### Tier 7 — Force/energy fixed-point review
- Force/energy scale 2³² is fine for all modes and **stays** (HMC/NUTS
  determinism depends on it). In double mode it caps force accumulation at ~32
  fractional bits — OpenMM accepts this; we will too.

### Tier 8 — Host/API + Python
- Host-side download/unscale code must read the right element size for
  `posq`/`velm`/double buffers.
- Python API stays `double` in/out (transparent — precision is an OpenMM context
  setting, nothing new to expose). ✅ This is a feature: zero Python API churn.

---

## Part D — Open questions for the user

1. **Knob surface** — where do the decoupled axes live in the API? Per-Force
   settings (e.g. `GridForce.setGridStoragePrecision(...)`,
   `setHessianPrecision(...)`), context-creation properties, or both? They must
   land in the `defines` map at `createModule` time either way.
2. **Default for grid storage** — follow `real` (memory-lean, OpenMM-consistent)
   vs. always f64 (accuracy-first). Leaning follow-`real` as the default with
   f64 opt-in.
3. ~~**Hessian accumulation**~~ — LARGELY RESOLVED by the probe: deterministic
   fixed-point, single 64-bit accumulator, scale 2²⁹–2³⁰ (up from 2²⁴). No wide
   accumulator needed. Remaining: confirm the metric-path guard is acceptable
   (vs. supporting a non-deterministic metric), and optionally re-run the probe
   on production poses before fixing the constant.
4. **Goal of `double` compute mode** — true bit-for-bit double, or just "correct
   + more accurate than single"? Sets how hard Tier 7 needs to push.
5. **Mixed mode for grid kernels** — do grid forces need `posqCorrection`
   compensated positions, or is float position fine for the grid path (it already
   is for the OpenMM nonbonded path)? Likely float position is fine; confirm.
6. **Templates vs. injected typedefs** — adopt OpenMM's injected `real`/`mixed`
   for compute (recommended, less code); `GRID_STORAGE_TYPE`/`HESSIAN_TYPE` are
   separate plain `#define`s. Confirm no kernel needs true template
   instantiation.

---

## Part E — Suggested phasing (once questions are answered)

- **Phase 0:** Add a regression harness — energies/forces/Hessians for a fixed
  system, captured in single mode *now* (golden values), so every later phase is
  diffable. Add a double-precision smoke test that currently *fails*/produces
  garbage to prove the latent bug, then tracks the fix.
- **Phase 1 (Tier 1):** posq/velm element-size correctness across all kernels.
  Fixes the double-mode bug. Self-contained, testable.
- **Phase 2 (Tier 5):** isolated-kernel cleanup — smallest, builds fluency with
  the `real`/`mixed` pattern in this codebase.
- **Phase 3 (Tier 2):** grid kernels + shared headers to `real`/`mixed`.
- **Phase 4 (Tier 3/4/6):** grid storage precision + GBSA reconciliation +
  fixed-point review — the design-heavy phase.
- **Phase 5:** docs, perf validation (single mode must not regress), VRAM
  validation for double/mixed.

## Part F — Risks
- **Blast radius via shared `.cuh` headers** — one change touches many kernels;
  needs the Phase 0 harness.
- **Silent numerical drift** — float→real in single mode should be a *no-op*;
  any diff signals an accidental double-promotion or intrinsic mismatch.
- **VRAM blowup** in double/mixed if grid storage follows precision (Tier 3).
- **Perf regression** in the hot single-precision path if intrinsics or
  accumulation types change inadvertently.
- **Determinism** of HMC/NUTS integrators depends on fixed-point summation —
  don't replace with float atomics.
</content>
</invoke>
