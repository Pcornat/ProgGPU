//
// Created by postaron on 26/02/2019.
//

#include "run_funtions.h"
#include <stdio.h>
#include <stdlib.h>
#include <utils.h>

int32_t run_openmp(size_t *matCol, size_t *matRow, float *step) {
	return EXIT_SUCCESS;
}

int32_t run_cuda(size_t *matCol, size_t *matRow, float *step) {
	return EXIT_SUCCESS;
}

int32_t run_sequentiel(size_t *matCol, size_t *matRow, float *step) {
	float mat[((*matCol) + 1) * ((*matRow) + 1)];


	return EXIT_SUCCESS;
}

int32_t run_config(const char *filename, size_t *matCol, size_t *matRow, float *step) {
	//Pas de vérif pour la config, flemme
	FILE *file = NULL;
	int res = 0;

	file = fopen(filename, "r");

	if (file == NULL)
		return EXIT_FAILURE;

	if (fscanf(file, "%zu", matCol) != 0 || fscanf(file, "%zu", matRow) != 0 || fscanf(file, "%f", step) != 0)
		return EXIT_FAILURE;

	return fclose(file);
}