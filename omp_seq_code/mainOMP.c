/* standard include */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <opencv2/core/core_c.h>

/* user include */
#include <run_functions.h>

int main(int argc, char *argv[]) {
	/*
	 * 2/3 arguments : fichierConf, flagOMP.
	 * Séquentiel par défaut si juste un seul argument.
	 */

	int32_t returnValue = 0, omp = 0;
	uint32_t sortieImage = 1;
	size_t matRow = 0, matCol = 0, numIter = 1, numHeatPoint = 0, sx = 1, sy = 1;
	float *matrix = NULL, newMatrix = NULL;
	float coeffD = 0.2f;
	const float convergence = 1e-3f;
	src_t *heatPoint = NULL;

	CvMat *img = NULL;

	if (argc == 2 || argc == 3) {
		if (!run_config(argv[1], matrix, newMatrix, &matCol, &matRow, &numIter, &coeffD, &sortieImage, heatPoint, &numHeatPoint))
			return EXIT_FAILURE;

		if (argc == 3) {
			omp = (int32_t) strtol(argv[3], NULL, 10);
		}
	} else {
		fprintf(stderr, "Utilisation : <FichierConfig> <FlagOMP>\n");
		fprintf(stderr, "\t<FichierConfig> : obligatoire. C'est un fichier texte qui permet de configurer le problème entièrement. + de détails dans le fichier.\n");
		fprintf(stderr, "\t<FlagOMP> : optionnelle. Par défaut, séquentielle. Si différent de 0, calcul parallèle avec OpenMP activé.\n");
		return EXIT_FAILURE;
	}

	if (matCol < 256 || matRow < 256) {
		sx = 256 / matCol;
		sy = 256 / matRow;
	}

	img = cvCreateMat(matRow * sy, matCol * sx, CV_8UC3);

	if (omp != 1) {
		returnValue = run_sequentiel(matrix, newMatrix, matCol, matRow, numIter, coeffD, sortieImage, heatPoint, numHeatPoint, convergence, img);
	} else {
		returnValue = run_openmp(matrix, newMatrix, matCol, matRow, numIter, coeffD, sortieImage, heatPoint, numHeatPoint, convergence, img);
	}

	end_simulation(matrix, newMatrix, heatPoint);
	cvReleaseData(img);

	return returnValue;
}