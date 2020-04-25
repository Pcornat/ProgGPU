/* System incldues */
#include <cstdio>
#include <cstdlib>
//#include <opencv2/core/core_c.h>


//#define CONVERGENCE 1e-3f
/* user includes */
#include "cuda_functions.cuh"
#include "compute_functions.cuh"


/*
 * args : fichierConf flagCuda/OMP
 */
int main(int argc, char *argv[]) {
	constexpr double CONVERGENCE = 1e-3f;
	int32_t returnValue;
	uint32_t numIter = 0, sortieImage = 1;

	size_t matCol = 0, matRow = 0,
			sx = 1, sy = 1,
			srcSize = 0;

	float *h_val = nullptr;

	heatPoint *h_srcs = nullptr;
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
	cv::Mat img{ static_cast<int>(matRow * sy), static_cast<int>(matCol * sx), CV_8UC3 };

//	img = cvCreateMat((int32_t) (matRow * sy), (int32_t) (matCol * sx), CV_8UC3);

	returnValue = run_cuda(h_val, matCol, matRow, h_srcs, srcSize, numIter, sortieImage, img, CONVERGENCE);

	return returnValue;
}