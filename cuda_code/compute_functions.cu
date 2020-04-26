//
// Created by postaron on 04/04/2019.
//

#include "compute_functions.cuh"

#ifdef HAVE_CUB

#include <cub/cub.cuh>

#endif

__device__ void keepHeat(float *__restrict d_val,
						 float *__restrict d_val_new,
						 const size_t m,
						 const size_t n,
						 const heatPoint *__restrict srcs,
						 const size_t numHeat,
						 const uint32_t x,
						 const uint32_t y) {
	for (size_t i = 0; i < numHeat; ++i) {
		if (srcs[i].x == x && srcs[i].y == y) {
			d_val[offset(srcs[i].x, srcs[i].y, m)] = 1.0f;
			d_val_new[offset(srcs[i].x, srcs[i].y, m)] = 1.0f;
		}
	}
}

__global__ void simulationKernel(float *__restrict d_val_new,
								 float *__restrict d_val,
								 const size_t m,
								 const size_t n,
								 const float convergence,
								 const uint32_t nite,
								 const heatPoint *__restrict const d_srcsHeat,
								 const size_t numHeat) {
	const std::uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x,
			iy = blockIdx.y * blockDim.y + threadIdx.y;

	float error = 1.0f;

	if (ix > 1 && ix < n && iy > 1 && iy < m) {
		for (size_t i = 0; i < nite && error > convergence; ++i) {
			d_val_new[offset(ix, iy, m)] = 0.25f *
										   (d_val[offset(ix, iy - 1, m)] +
											d_val[offset(ix, iy + 1, m)] +
											d_val[offset(ix - 1, iy, m)] +
											d_val[offset(ix + 1, iy, m)]);

			error = fmaxf(error, fabsf(d_val_new[offset(ix, iy, m)] - d_val[offset(ix, iy, m)]));

			d_val[offset(ix, iy, m)] = d_val_new[offset(ix, iy, m)];

			keepHeat(d_val, d_val_new, m, n, d_srcsHeat, numHeat, ix, iy);
		}
	}

}

