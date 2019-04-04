//
// Created by postaron on 26/02/2019.
//

#ifndef PROGGPU_CUDA_FUNCTIONS_CUH
#define PROGGPU_CUDA_FUNCTIONS_CUH

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <opencv2/core/core_c.h>
#include "images.cuh"
#include "compute_functions.cuh"

bool run_configCUDA(const char *filename, float **matrix, size_t *matCol, size_t *matRow, heatPoint **srcsHeat, size_t *srcsSize, uint32_t *numIter, uint32_t *sortieImage);

int32_t run_cuda(float *h_matrix, size_t matCol, size_t matRow, heatPoint *h_srcs, size_t srcSize, uint32_t numIter, uint32_t sortieImage, CvMat *img, float convergence);

void end_simulation(float *__restrict h_matrix, heatPoint *__restrict h_srcs);

#endif //PROGGPU_CUDA_FUNCTIONS_CUH
