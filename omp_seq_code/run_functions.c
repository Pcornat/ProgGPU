//
// Created by postaron on 26/02/2019.
//

#include "run_functions.h"
#include <stdio.h>

bool run_config(const char *filename, float **matrix, float **newMatrix, size_t *matCol, size_t *matRow, uint32_t *numIter, uint32_t *sortieImage) {
	//Vérif partielle. Je pense qu'elle est partielle. À voir sinon.
	int64_t numHeatPnt = 0;
	FILE *file = NULL;
	file = fopen(filename, "r");

	if (file == NULL)
		return false;

	if (fscanf(file, "%zu", matCol) == EOF || fscanf(file, "%zu", matRow) == EOF || fscanf(file, "%u", numIter) == EOF || fscanf(file, "%u", sortieImage) == EOF ||
		fscanf(file, "%li", &numHeatPnt) == EOF) {
		fclose(file);
		return false;
	}

	if (numHeatPnt <= 0) {
		fclose(file);
		return false;
	}

	if (*matCol <= 0) *matCol = 1000;
	if (*matRow <= 0) *matRow = 1000;

	if ((*matrix = (float *) calloc(*matRow * *matCol, sizeof(float))) == NULL || (*newMatrix = (float *) calloc(*matRow * *matCol, sizeof(float))) == NULL) {
		fclose(file);
		return false;
	}

	if (*sortieImage > *numIter)
		*sortieImage %= *numIter;

	for (int64_t i = 0; i < numHeatPnt; ++i) {
		int64_t m = 0, n = 0;
		if (fscanf(file, "%li", &m) == EOF || fscanf(file, "%li", &n) == EOF) {
			fclose(file);
			free(matrix);
			free(newMatrix);
			return false;
		}
		if (m < 0 || m >= *matRow || n < 0 || n >= *matCol) {
			fclose(file);
			free(matrix);
			free(newMatrix);
			return false;
		}
		(*matrix)[m * *matRow + n] = 1.0f;
		(*newMatrix)[m * *matRow + n] = 1.0f;
	}

	return fclose(file) == 0;
}

void end_simulation(float *restrict matrix, float *restrict newMatrix) {
	puts("Libération mémoire.");
	free(newMatrix);
	free(matrix);
}
