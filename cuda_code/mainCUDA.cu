#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <structures.h>
#include <cuda_functions.h>


/*
 * args : fichierConf flagCuda/OMP
 */
int main(int argc, char *argv[]) {
	char *fichierConf = NULL;
	size_t matCol = 0, matRow = 0;
	int32_t returnValue;
	float step = 0.f, degre = 0.f, coeffD = 0.f;
	point heatPoint = {0, 0};

	/*
	 * Au moins 1 arguments.
	 */
	if (argc != 2) {
		fprintf(stderr, "Erreur d'arguments : ./prog fichier.conf\n");
		return EXIT_FAILURE;
	}

	fichierConf = argv[1];

	if (!run_config(argv[2], &matCol, &matRow, &step, &degre, &coeffD, &heatPoint))
		return EXIT_FAILURE;

	returnValue = run_cuda(&matCol, &matRow, &step, &degre, &coeffD, &heatPoint);

	return returnValue;
}