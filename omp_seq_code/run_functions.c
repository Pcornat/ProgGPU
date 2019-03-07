//
// Created by postaron on 26/02/2019.
//

#include "run_functions.h"

int32_t
run_openmp(float *matrix, float *newMatrix, size_t matCol, size_t matRow, size_t numIter, const float coeffD, uint32_t sortieImage, src_t *heatPoint, size_t numHeatPoint, const float convergence,
		   CvMat *img) {

	return EXIT_SUCCESS;
}

int32_t
run_sequentiel(float *matrix, float *newMatrix, size_t matCol, size_t matRow, size_t numIter, const float coeffD, uint32_t sortieImage, src_t *heatPoint, size_t numHeatPoint, const float convergence,
			   CvMat *img) {
	return EXIT_SUCCESS;
}

bool run_config(const char *filename, float *matrix, float *newMatrix, size_t *matCol, size_t *matRow, size_t *numIter, float *coeffD, uint32_t *sortieImage, src_t *heatPoint, size_t *numHeatPoint) {
	//Vérif partielle. Je pense qu'elle est partielle. À voir sinon.
	int64_t nbrHeatPtTmp = 0;
	FILE *file = NULL;
	file = fopen(filename, "r");

	if (file == NULL)
		return false;

	if (fscanf(file, "%zu", matCol) != 0 || fscanf(file, "%zu", matRow) != 0 || fscanf(file, "%zu", numIter) != 0 || fscanf(file, "%u", sortieImage) != 0 || fscanf(file, "%lu", &nbrHeatPtTmp) != 0) {
		fclose(file);
		return false;
	}

	if (*matCol <= 0) *matCol = 1000;
	if (*matRow <= 0) *matRow = 1000;

	if ((matrix = (float *) calloc(*matRow * *matCol, sizeof(float))) == NULL || (newMatrix = (float *) calloc(*matRow * *matCol, sizeof(float))) == NULL) {
		fclose(file);
		return false;
	}

	if (*sortieImage > *numIter)
		*sortieImage %= *numIter;

	if ((nbrHeatPtTmp <= 0) || (nbrHeatPtTmp > (*matCol * *matRow)))
		*numHeatPoint = 1;

	if ((heatPoint = (src_t *) malloc(*numHeatPoint * sizeof(src_t))) == NULL) {
		free(newMatrix);
		free(matrix);
		fclose(file);
		return false;
	}

	for (size_t i = 0; i < *numHeatPoint; ++i) {
		uint32_t x = 0, y = 0;
		float t = 0.f;

		if (fscanf(file, "%u", &x) != 0 || fscanf(file, "%u", &y) != 0 || fscanf(file, "%f", &t) != 0) {
			free(newMatrix);
			free(matrix);
			free(heatPoint);
			fclose(file);
			return false;
		}

		if (x > *matCol || y > *matRow) {
			free(newMatrix);
			free(matrix);
			free(heatPoint);
			fclose(file);
			return false;
		}

		if (t <= 0.f)
			t = 100.f;

		heatPoint[i].x = x;
		heatPoint[i].y = y;
		heatPoint[i].t = t;
	}

	return fclose(file) == 0;
}

void end_simulation(float *matrix, float *newMatrix, src_t *heatPoint) {
	puts("Libération mémoire.");
	free(newMatrix);
	free(matrix);
	free(heatPoint);
}
