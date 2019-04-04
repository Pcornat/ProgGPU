/* standard include */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <opencv2/core/core_c.h>

/* user include */
#include "run_functions.h"
#include "compute_functions.h"

int main(int argc, char *argv[]) {
	/*
	 * 1 argument : fichierConf
	 */

	int32_t returnValue = 0;
	uint32_t sortieImage = 1, numIter = 1;
	size_t matRow = 0, matCol = 0, sx = 1, sy = 1, srcSize = 0;
	float *matrix = NULL, *newMatrix = NULL;
	heatPoint *srcs = NULL;
	const float convergence = 1e-3f;

	CvMat *img = NULL;

	if (argc == 2) {
		if (!run_config(argv[1], &matrix, &newMatrix, &matCol, &matRow, &numIter, &sortieImage, &srcs, &srcSize))
			return EXIT_FAILURE;
	} else {
		fprintf(stderr, "Utilisation : <FichierConfig>\n");
		fprintf(stderr, "\t<FichierConfig> : obligatoire. C'est un fichier texte qui permet de configurer le problème entièrement. + de détails dans le fichier.\n");
		return EXIT_FAILURE;
	}

	if (matCol < 256 || matRow < 256) {
		sx = 256 / matCol;
		sy = 256 / matRow;
	}

	img = cvCreateMat((int32_t) (matRow * sy), (int32_t) (matCol * sx), CV_8UC3);

	simulation(matrix, newMatrix, matRow, matCol, convergence, numIter, sortieImage, img, srcs, srcSize);

	end_simulation(matrix, newMatrix, srcs);
	cvReleaseData(img);

	return returnValue;
}