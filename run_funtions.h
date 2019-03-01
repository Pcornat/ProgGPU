//
// Created by postaron on 26/02/2019.
//

#ifndef PROGGPU_RUN_FUNTIONS_H
#define PROGGPU_RUN_FUNTIONS_H

#include <stddef.h>
#include <stdint.h>

int32_t run_config(const char *filename, size_t *matCol, size_t *matRow, float *step);

int32_t run_openmp(size_t *matCol, size_t *matRow, float *step);

int32_t run_cuda(size_t *matCol, size_t *matRow, float *step);

int32_t run_sequentiel(size_t *matCol, size_t *matRow, float *step);

#endif //PROGGPU_RUN_FUNTIONS_H