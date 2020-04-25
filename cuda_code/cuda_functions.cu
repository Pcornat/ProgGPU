#include "cuda_functions.cuh"
#include "compute_functions.cuh"
#include <cstdio>
#include <cmath>
#include <cuda.h>

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

bool run_configCUDA(const char *filename,
					float **matrix,
					size_t *matCol,
					size_t *matRow,
					heatPoint **srcsHeat,
					size_t *srcsSize,
					uint32_t *numIter,
					uint32_t *sortieImage) {
	//Vérif partielle.
	int64_t numHeatPnt = 0;
	FILE *file = NULL;
	file = fopen(filename, "r");

	if (file == NULL) {
		perror("Erreur fopen : ");
		return false;
	}

	if (fscanf(file, "%zu", matCol) == EOF ||
		fscanf(file, "%zu", matRow) == EOF ||
		fscanf(file, "%u", numIter) == EOF ||
		fscanf(file, "%u", sortieImage) == EOF ||
		fscanf(file, "%li", &numHeatPnt) == EOF) {
		perror("Erreur fscanf : ");
		fclose(file);
		return false;
	}

	if (numHeatPnt <= 0) {
		fclose(file);
		return false;
	}

	*srcsSize = numHeatPnt;

	if (*matCol <= 0) *matCol = 1000;
	if (*matRow <= 0) *matRow = 1000;

	if ((*matrix = (float *) calloc(*matRow * *matCol, sizeof(float))) == NULL) {
		perror("Erreir calloc matrices : ");
		fclose(file);
		return false;
	}

	if ((*srcsHeat = (heatPoint *) calloc(*srcsSize, sizeof(heatPoint))) == NULL) {
		perror("Error malloc heatPoints :");
		free(*matrix);
		fclose(file);
		return false;
	}

	if (*sortieImage > *numIter)
		*sortieImage %= *numIter;

	for (int64_t i = 0; i < numHeatPnt; ++i) {
		int64_t m = 0, n = 0;
		size_t x, y;
		if (fscanf(file, "%li", &m) == EOF || fscanf(file, "%li", &n) == EOF) {
			fclose(file);
			free(matrix);
			return false;
		}
		if (m < 0 || m >= *matRow || n < 0 || n >= *matCol) {
			fclose(file);
			free(matrix);
			return false;
		}
		(*srcsHeat)[i].x = x = (size_t) n, (*srcsHeat)[i].y = y = (size_t) m;
		/*
		 * Les coordonnées données dans le fichier de configuration servent à décrire le milieu du point de chaleur (c'est un carré)
		 */
		(*matrix)[offset(x, y, *matRow)] = 1.0f;

		(*matrix)[offset(x, y + 1, *matRow)] = 1.0f;

		(*matrix)[offset(x + 1, y, *matRow)] = 1.0f;

		(*matrix)[offset(x + 1, y + 1, *matRow)] = 1.0f;

		(*matrix)[offset(x - 1, y, *matRow)] = 1.0f;

		(*matrix)[offset(x, y - 1, *matRow)] = 1.0f;

		(*matrix)[offset(x - 1, y - 1, *matRow)] = 1.0f;

		(*matrix)[offset(x - 1, y + 1, *matRow)] = 1.0f;

		(*matrix)[offset(x + 1, y - 1, *matRow)] = 1.0f;

	}

	return fclose(file) == 0;
}

int32_t run_cuda(float *h_matrix,
				 size_t matCol,
				 size_t matRow,
				 heatPoint *h_srcs,
				 size_t srcSize,
				 uint32_t numIter,
				 uint32_t sortieImage,
				 CvMat *img,
				 float convergence) {
	uint32_t numThread = 16;
	float *d_val = NULL, *d_val_new = NULL, kernelTime = 0.f;
	heatPoint *d_srcs = NULL;
	cudaEvent_t start, stop;

	cudaEventCreate(&start), cudaEventCreate(&stop);

	if (cudaMemChk(cudaMalloc((void **) &d_val, matCol * matRow * sizeof(float))) == EXIT_FAILURE) {
		end_simulation(h_matrix, h_srcs);
		return EXIT_FAILURE;
	}

	if (cudaMemChk(cudaMalloc((void **) &d_val_new, matCol * matRow * sizeof(float))) == EXIT_FAILURE) {
		cudaFree(d_val);
		end_simulation(h_matrix, h_srcs);
		return EXIT_FAILURE;
	}

	if (cudaMemChk(cudaMalloc((void **) &d_srcs, srcSize * sizeof(heatPoint))) == EXIT_FAILURE) {
		cudaFree(d_val);
		cudaFree(d_val_new);
		end_simulation(h_matrix, h_srcs);
		return EXIT_FAILURE;
	}

	if (cudaMemChk(cudaMemcpy(d_val, h_matrix, matCol * matRow * sizeof(float), cudaMemcpyHostToDevice)) == EXIT_FAILURE) {
		cudaFree(d_val);
		cudaFree(d_val_new);
		cudaFree(d_srcs);
		end_simulation(h_matrix, h_srcs);
		return EXIT_FAILURE;
	}

	if (cudaMemChk(cudaMemcpy(d_srcs, h_srcs, srcSize * sizeof(heatPoint), cudaMemcpyHostToDevice)) == EXIT_FAILURE) {
		cudaFree(d_val);
		cudaFree(d_val_new);
		cudaFree(d_srcs);
		end_simulation(h_matrix, h_srcs);
		return EXIT_FAILURE;
	}

	if (cudaMemChk(cudaMemset(d_val_new, 0, matCol * matRow * sizeof(float))) == EXIT_FAILURE) {
		cudaFree(d_val);
		cudaFree(d_val_new);
		cudaFree(d_srcs);
		end_simulation(h_matrix, h_srcs);
		return EXIT_FAILURE;
	}

	//À optimiser avec les defines en fonction du GPU (en dur pour le Kepler pour l'instant)
	dim3 dimGrid;
	dimGrid.x = (uint32_t) ceil((matCol - 1.0) / numThread);
	dimGrid.y = (uint32_t) ceil((matRow - 1.0) / numThread);
	dimGrid.z = 1;

	dim3 dimBlock;
	dimBlock.x = numThread;
	dimBlock.y = numThread;
	dimBlock.z = 1;

	//Lancement du chrono
	cudaEventRecord(start);
	simulationKernel << < dimGrid, dimBlock >> > (d_val_new, d_val, matCol, matRow, convergence, numIter, d_srcs, srcSize);
	cudaEventRecord(stop); //Arrêt

	if (cudaMemChk(cudaMemcpy(h_matrix, d_val, matCol * matRow * sizeof(float), cudaMemcpyDeviceToHost)) == EXIT_FAILURE) {
		fprintf(stderr, "Transfert du résultat impossible.\n");
		cudaFree(d_val);
		cudaFree(d_val_new);
		cudaFree(d_srcs);
		end_simulation(h_matrix, h_srcs);
		return EXIT_SUCCESS;
	}
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&kernelTime, start, stop);

	printf("Temps de la simulation : %f\n", kernelTime);

	cudaFree(d_val);
	cudaFree(d_val_new);
	cudaFree(d_srcs);
	end_simulation(h_matrix, h_srcs);
	return EXIT_SUCCESS;
}

void end_simulation(float *__restrict h_matrix, heatPoint *__restrict h_srcs) {
	free(h_matrix);
	free(h_srcs);
	puts("Libération mémoire");
}