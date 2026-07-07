// Software atomicAdd(double*, double) for pre-sm_60 GPUs (Maxwell etc).
// Hardware double-atomic was added in sm_60 (Pascal); on older arches the
// NVIDIA-recommended atomicCAS loop is used. Defined inline only when the
// compile target lacks the hardware overload, so the symbol is harmless on
// newer arches.

#ifndef ATOMICADD_DOUBLE_CUH
#define ATOMICADD_DOUBLE_CUH

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ static inline double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#endif // ATOMICADD_DOUBLE_CUH
