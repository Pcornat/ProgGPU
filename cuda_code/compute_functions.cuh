#ifndef PROGGPU_COMPUTE_FUNCTIONS_CUH
#define PROGGPU_COMPUTE_FUNCTIONS_CUH

#include <cstdint>

struct heatPoint {
	size_t x;
	size_t y;
};

[[gnu::always_inline]] inline __host__ __device__
size_t offset(size_t x, size_t y, size_t m) {
	return x * m + y;
}

__device__ void keepHeat(float *__restrict d_val,
						 float *__restrict d_val_new,
						 const size_t m,
						 const size_t n,
						 const heatPoint *__restrict srcs,
						 const size_t numHeat,
						 const uint32_t x,
						 const uint32_t y);

__global__ void simulationKernel(float *__restrict d_val_new,
								 float *__restrict d_val,
								 const size_t m,
								 const size_t n,
								 const float convergence,
								 const uint32_t nite,
								 const heatPoint *__restrict const d_srcsHeat,
								 const size_t numHeat);

#endif //PROGGPU_COMPUTE_FUNCTIONS_CUH
