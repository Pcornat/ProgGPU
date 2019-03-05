//
// Created by postaron on 26/02/2019.
//

#ifndef PROGGPU_RUN_FUNTIONS_H
#define PROGGPU_RUN_FUNTIONS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "structures.h"

bool run_config(const char *filename, size_t *matCol, size_t *matRow, float *step, float *degre, float *coeffD, point *heatPoint);

int32_t run_openmp(size_t *matCol, size_t *matRow, float *step, float *degre, float *coeffD, point *heatPoint);

int32_t run_sequentiel(size_t *matCol, size_t *matRow, float *step, float *degre, float *coeffD, point *heatPoint);

#endif //PROGGPU_RUN_FUNTIONS_H