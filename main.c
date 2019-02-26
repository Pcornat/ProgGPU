#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <run_funtions.h>

int main(int argc, char *argv[]) {
	bool flagCuda = false;
	bool sequentiel = true;

	/*
	 * Arguments.
	 * Si aucun argument : séquentiel
	 * Si un argument à 1 : CUDA
	 * Si un argument à 0 : OpenMP
	 */
	switch (argc) {
		case 2:
			int arg = atoi(argv[1]);
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

		default:
			puts("Code sequentiel lance.");
			break;
	}

	if (flagCuda && !sequentiel)
		run_cuda();
	else if (!flagCuda && !sequentiel)
		run_openmp();
	else
		run_sequentiel();

	return EXIT_SUCCESS;
}