/* System incldues */
#include <cstdio>
#include <cstdlib>


//#define CONVERGENCE 1e-3f
/* user includes */
#include "cuda_functions.cuh"
#include "compute_functions.cuh"


/*
 * args : fichierConf flagCuda/OMP
 */
int main(int argc, char *argv[]) {
	constexpr double CONVERGENCE = 1e-3f;
	uint32_t numIter = 0, sortieImage = 1;

	size_t matCol = 0, matRow = 0,
			sx = 1, sy = 1,
			srcSize = 0;

	host_vector<float> h_val;

	host_vector<heatPoint> h_srcs;
	/*
	 * Au moins 1 arguments.
	 */
	if (argc != 2) {
		std::cerr << "Erreur d'arguments : ./prog fichier.conf\n";
		return EXIT_FAILURE;
	}

	run_configCUDA(argv[1], h_val, matCol, matRow, h_srcs, numIter, sortieImage);

	if (matCol < 256 || matRow < 256) {
		sx = 256 / matCol;
		sy = 256 / matRow;
	}
	cv::Mat img{ static_cast<int>(matRow * sy), static_cast<int>(matCol * sx), CV_8UC3 };

	run_cuda(h_val, matCol, matRow, h_srcs, srcSize, numIter, sortieImage, img, CONVERGENCE);

	return EXIT_SUCCESS;
}