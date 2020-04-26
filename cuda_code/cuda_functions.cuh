//
// Created by postaron on 26/02/2019.
//

#ifndef PROGGPU_CUDA_FUNCTIONS_CUH
#define PROGGPU_CUDA_FUNCTIONS_CUH

#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include "compute_functions.cuh"
#include <thrust/host_vector.h>

template<typename T>
using host_vector = thrust::host_vector<T>;

void run_configCUDA(const char *filename,
					host_vector<float> &matrix,
					size_t &matCol,
					size_t &matRow,
					host_vector<heatPoint> &srcsHeat,
					uint32_t &numIter,
					uint32_t &sortieImage);

void run_cuda(host_vector<float> &h_matrix,
			  size_t matCol,
			  size_t matRow,
			  host_vector<heatPoint> &h_srcs,
			  size_t srcSize,
			  uint32_t numIter,
			  uint32_t sortieImage,
			  cv::Mat &img,
			  const float convergence);


#endif //PROGGPU_CUDA_FUNCTIONS_CUH
