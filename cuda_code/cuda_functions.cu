// Created by postaron on 26/02/2019.
//

#include "cuda_functions.h"
#include <stdio.h>

bool run_config(const char *filename, size_t *matCol, size_t *matRow, float *step, float *degre, float *coeffD, point *heatPoint) {
	//Pas de vérif pour la config, flemme
	FILE *file = NULL;
	file = fopen(filename, "r");

	if (file == NULL)
		return false;

	if (fscanf(file, "%zu", matCol) != 0 || fscanf(file, "%zu", matRow) != 0 || fscanf(file, "%f", step) != 0 || fscanf(file, "%u", &heatPoint->x) != 0 || fscanf(file, "%u", &heatPoint->y) != 0 ||
		fscanf(file, "%f", degre) != 0 || fscanf(file, "%f", coeffD) != 0)
		return false;

	if (fclose(file) != 0)
		return false;
	return true;
}

bool run_cuda(size_t *matCol, size_t *matRow, float *step, float *degre, float *coeffD, point *heatPoint) {
	return true;
}