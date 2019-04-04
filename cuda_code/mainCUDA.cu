/* System incldues */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <opencv2/core/core_c.h>


#define CONVERGENCE 1e-3f
/* user includes */
#include "cuda_functions.cuh"
#include "compute_functions.cuh"


/*
 * args : fichierConf flagCuda/OMP
 */
int main(int argc, char *argv[]) {
	int32_t returnValue;
	uint32_t numIter = 0, sortieImage = 1;

	size_t matCol = 0, matRow = 0,
			sx = 1, sy = 1,
			srcSize = 0;

	float *h_val = NULL;

	heatPoint *h_srcs = NULL;

	CvMat *img = NULL;

	/*
	 * Au moins 1 arguments.
	 */
	if (argc != 2) {
		fprintf(stderr, "Erreur d'arguments : ./prog fichier.conf\n");
		return EXIT_FAILURE;
	}

	if (!run_configCUDA(argv[1], &h_val, &matCol, &matRow, &h_srcs, &srcSize, &numIter, &sortieImage))
		return EXIT_FAILURE;

	if (matCol < 256 || matRow < 256) {
		sx = 256 / matCol;
		sy = 256 / matRow;
	}

	img = cvCreateMat((int32_t) (matRow * sy), (int32_t) (matCol * sx), CV_8UC3);

	returnValue = run_cuda(h_val, matCol, matRow, h_srcs, srcSize, numIter, sortieImage, img, 0);


	cvReleaseData(img);

	return returnValue;
}