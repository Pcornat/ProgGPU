//
// Created by postaron on 26/02/2019.
//

#ifndef PROGGPU_CUDA_FUNCTIONS_H
#define PROGGPU_CUDA_FUNCTIONS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <structures.h>

bool run_configCUDA(const char *filename, size_t *matCol, size_t *matRow, float *step, float *degre, float *coeffD, point *heatPoint);

bool run_cuda(size_t *matCol, size_t *matRow, float *step, float *degre, float *coeffD, point *heatPoint);

#endif //PROGGPU_CUDA_FUNCTIONS_H
