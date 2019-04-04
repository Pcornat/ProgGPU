//
// Created by postaron on 04/04/2019.
//

#include "compute_functions.cuh"
#include <cuda.h>
#include <device_launch_parameters.h>

__host__ __device__ size_t offset(size_t x, size_t y, size_t m) {
	return x * m + y;
}

__device__ float calcul(float *d_val_new, float *d_val, size_t m, size_t n, int32_t x, int32_t y) {
	float error = 0.0f;

	d_val_new[offset(x, y, m)] = 0.25 * (d_val[offset(x, y - 1, m)] + d_val[offset(x, y - 1, m)] + d_val[offset(x - 1, y, m)] + d_val[offset(x + 1, y, m)]);

	error = fmaxf(error, fabsf(d_val_new[offset(x, y, m)] - d_val[offset(x, y, m)]));

	return error;
}

__device__ void swap(float *d_val, const float *d_val_new, size_t m, size_t n, int32_t x, int32_t y) {
	d_val[offset(x, y, m)] = d_val_new[offset(x, y, m)];
}

__device__ void keepHeat(float *__restrict d_val, float *__restrict d_val_new, size_t m, size_t n, const heatPoint *__restrict srcs, size_t numHeat, int32_t x, int32_t y) {
	/*for (size_t i = 0; i < numHeat; ++i) {
		d_val[offset(srcs[i].x, srcs[i].y, m)] = 1.0f;
		d_val_new[offset(srcs[i].x, srcs[i].y, m)] = 1.0f;
	}*/
}

__global__ void simulationKernel(float *__restrict d_val_new, float *__restrict d_val, size_t m, size_t n, float convergence, uint32_t nite, const heatPoint *__restrict d_srcsHeat, size_t numHeat) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x,
			y = blockIdx.y * blockDim.y + threadIdx.y;

	float error = 1.0f;

	if (x > 0 && x < n && y > 0 && y < m) {
		for (size_t i = 0; i < nite && error > convergence; ++i) {
			error = calcul(d_val_new, d_val, m, n, x, y);

			swap(d_val, d_val_new, m, n, x, y);

			keepHeat(d_val, d_val_new, m, n, d_srcsHeat, numHeat, x, y);
		}
	}

}
