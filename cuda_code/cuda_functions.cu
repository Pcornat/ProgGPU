#include "cuda_functions.cuh"
#include "compute_functions.cuh"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <thrust/device_vector.h>

#define CUDA_RT_CALL(call) { \
cudaError_t cudaStatus = call; \
        if (cudaSuccess != cudaStatus) {\
            fprintf(stderr,\
            "ERROR: CUDA RT call \"%s\" in line %d of file %s failed"\
                    "with "\
                    "%s (%d).\n",\
                    #call, __LINE__, __FILE__,cudaGetErrorString(cudaStatus), cudaStatus); \
            throw std::runtime_error("CUDA ERROR"); \
        }\
    }


inline int32_t cudaMemChk(cudaError_t error) {
	if (error != cudaSuccess) {
		fprintf(stderr, "Erreur allocation CUDA\n");
		return EXIT_FAILURE;
	} else
		return EXIT_SUCCESS;
}

void run_configCUDA(const char *filename,
					host_vector<float> &matrix,
					size_t &matCol,
					size_t &matRow,
					host_vector<heatPoint> &srcsHeat,
					uint32_t &numIter,
					uint32_t &sortieImage) {
	//Vérif partielle.
	int64_t numHeatPnt = 0;
	std::ifstream file(filename);

	file >> matCol >> matRow >> numIter >> sortieImage >> numHeatPnt;

	if (numHeatPnt <= 0) {
		throw std::runtime_error("Au moins 1 point de chaleur nécessaire.");
	}


	if (matCol <= 0) matCol = 1000;
	if (matRow <= 0) matRow = 1000;

	matrix.resize(matRow * matCol, 0.f);

	srcsHeat.resize(numHeatPnt, { 0, 0 });

	if (sortieImage > numIter)
		sortieImage %= numIter;

	for (auto &heat : srcsHeat) {
		int64_t m = 0, n = 0;
		size_t x, y;

		file >> m >> n;
		if (m < 0 || m >= matRow || n < 0 || n >= matCol) {
			throw std::runtime_error("Les coordonnées du point de chaleur ne sont pas dans la matrice.");
		}
		heat.x = x = static_cast<size_t>(n), heat.y = y = static_cast<size_t>(m);
		/*
		 * Les coordonnées données dans le fichier de configuration servent à décrire le milieu du point de chaleur (c'est un carré)
		 */
		matrix[offset(x, y, matRow)] = 1.0f;

		matrix[offset(x, y + 1, matRow)] = 1.0f;

		matrix[offset(x + 1, y, matRow)] = 1.0f;

		matrix[offset(x + 1, y + 1, matRow)] = 1.0f;

		matrix[offset(x - 1, y, matRow)] = 1.0f;

		matrix[offset(x, y - 1, matRow)] = 1.0f;

		matrix[offset(x - 1, y - 1, matRow)] = 1.0f;

		matrix[offset(x - 1, y + 1, matRow)] = 1.0f;

		matrix[offset(x + 1, y - 1, matRow)] = 1.0f;

	}
}

void run_cuda(host_vector<float> &h_matrix,
			  size_t matCol,
			  size_t matRow,
			  host_vector<heatPoint> &h_srcs,
			  size_t srcSize,
			  uint32_t numIter,
			  uint32_t sortieImage,
			  cv::Mat &img,
			  const float convergence) {
	constexpr uint32_t numThread = 256;
	thrust::device_vector<float> d_val, d_val_new;
	thrust::device_vector<heatPoint> d_srcs;
	float kernelTime = 0.f;
	cudaEvent_t start, stop;

	cudaEventCreate(&start), cudaEventCreate(&stop);
	d_val.resize(matCol * matRow);
	d_val_new.resize(matCol * matRow, 0);
	d_srcs.resize(h_srcs.size());

	//À optimiser avec les defines en fonction du GPU (en dur pour le Kepler pour l'instant)
	const dim3 dimGrid{
			static_cast<std::uint32_t>(std::ceil((matCol - 1.0) / numThread)),
			static_cast<std::uint32_t>(std::ceil((matRow - 1.0) / numThread)),
			1 };

	const dim3 dimBlock{ numThread, numThread, 1 };
	CUDA_RT_CALL(cudaMemcpyAsync(thrust::raw_pointer_cast(d_val.data()),
								 thrust::raw_pointer_cast(h_matrix.data()),
								 h_matrix.size() * sizeof(float),
								 cudaMemcpyHostToDevice))
	CUDA_RT_CALL(cudaMemcpyAsync(thrust::raw_pointer_cast(d_srcs.data()),
								 thrust::raw_pointer_cast(h_srcs.data()),
								 h_srcs.size() * sizeof(heatPoint),
								 cudaMemcpyHostToDevice))
	//Lancement du chrono
	cudaEventRecord(start);
	simulationKernel<<< dimGrid, dimBlock >>>(d_val_new.data().get(),
											  d_val.data().get(),
											  matCol,
											  matRow,
											  convergence,
											  numIter,
											  d_srcs.data().get(),
											  srcSize);
	cudaEventRecord(stop); //Arrêt
	cudaEventSynchronize(stop);

	CUDA_RT_CALL(cudaMemcpyAsync(thrust::raw_pointer_cast(h_matrix.data()),
								 thrust::raw_pointer_cast(d_val.data()),
								 h_matrix.size() * sizeof(float),
								 cudaMemcpyDeviceToHost))
	CUDA_RT_CALL(cudaMemcpyAsync(thrust::raw_pointer_cast(h_srcs.data()),
								 thrust::raw_pointer_cast(d_srcs.data()),
								 h_srcs.size() * sizeof(float),
								 cudaMemcpyDeviceToHost))

	cudaEventElapsedTime(&kernelTime, start, stop);

	printf("Temps de la simulation : %f\n", kernelTime);
}