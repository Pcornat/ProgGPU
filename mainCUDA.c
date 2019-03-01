#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <run_funtions.h>

/*
 * args : fichierConf flagCuda/OMP
 */
int main(int argc, char *argv[]) {
	bool flagCuda = false;
	bool sequentiel = true;

	size_t matCol = 0, matRow = 0;
	uint32_t returnValue, arg;
	float step = 0.f;

	/*
	 * Arguments.
	 * Si aucun argument : séquentiel
	 * Si un argument à 1 : CUDA
	 * Si un argument à 0 : OpenMP
	 */
	switch (argc) {
		case 3:
			//int32_t arg;
			arg = atoi(argv[3]);
			flagCuda = (arg == 0) ? true : false;
			sequentiel = false;
			switch (flagCuda) {
				case true:
					puts("Code en CUDA lance.");
					break;
				case false:
					puts("Code en OpenMP lancé");
					break;
				default:
					return EXIT_FAILURE;
			}
			break;

		case 2:
			puts("Code sequentiel lance.");
			break;

		default:
			puts("Cassé :D");
			break;
	}

	if (run_config(argv[2], &matCol, &matRow, &step) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (flagCuda && !sequentiel)
		returnValue = run_cuda(&matCol, &matRow, &step);
	else if (!flagCuda && !sequentiel)
		returnValue = run_openmp(&matCol, &matRow, &step);
	else
		returnValue = run_sequentiel(&matCol, &matRow, &step);

	return returnValue;
}