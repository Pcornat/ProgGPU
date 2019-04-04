//
// Created by postaron on 04/04/2019.
//

#ifndef PROGGPU_COMPUTE_FUNCTIONS_CUH
#define PROGGPU_COMPUTE_FUNCTIONS_CUH

#include <stddef.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <opencv2/core/core_c.h>

typedef struct {
	size_t x;
	size_t y;
} heatPoint;

__host__ __device__ size_t offset(size_t x, size_t y, size_t m);

__device__ float calcul(float *d_val_new, float *d_val, size_t m, size_t n, int32_t x, int32_t y);

__device__ void swap(float *d_val, const float *d_val_new, size_t m, size_t n, int32_t x, int32_t y);

__device__ void keepHeat(float *__restrict d_val, float *__restrict d_val_new, size_t m, size_t n, const heatPoint *__restrict srcs, size_t numHeat, int32_t x, int32_t y);

__global__ void simulationKernel(float *__restrict d_val_new, float *__restrict d_val, size_t m, size_t n, float convergence, uint32_t nite, const heatPoint *__restrict d_srcsHeat, size_t numHeat);

#endif //PROGGPU_COMPUTE_FUNCTIONS_CUH
