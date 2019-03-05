/* standard include */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

/* user include */
#include <omp_seq_code/run_functions.h>
#include <structures.h>

int main(int argc, char *argv[]) {
	/*
	 * 2 arguments : fichier de conf + flag omp = 1 sinon séquentiel.
	 * Séquentiel par défaut si juste un seul argument.
	 */
	bool returnValue = false;
	int32_t omp = 0;
	size_t matRow = 0, matCol = 0;
	float step = 0.f, degre = 0.f, coeffD = 0.f;
	point heatPoint = {0, 0};

	if (argc >= 2) {
		if (!run_config(argv[2], &matCol, &matRow, &step, &degre, &coeffD, &heatPoint)) {
			fprintf(stderr, "Erreur pour la lecture de la configuration.\n");
			return EXIT_FAILURE;
		}

		if (argc == 3) {
			omp = atoi(argv[3]);
		}
	} else {
		fprintf(stderr, "Erreur d'arguments : fichier_config flagOmp");
		return EXIT_FAILURE;
	}

	if (omp != 1) {
		returnValue = run_sequentiel(&matCol, &matRow, &step, &degre, &coeffD, &heatPoint);
	} else {
		returnValue = run_openmp(&matCol, &matRow, &step, &degre, &coeffD, &heatPoint);
	}

	switch (returnValue) {
		case true:
			return EXIT_SUCCESS;

		case false:
			return EXIT_FAILURE;

		default:
			return EXIT_FAILURE;
	}
}