#include "compute_functions.h"
#include <stdio.h>
#include <math.h>

extern inline size_t offset(size_t x, size_t y, size_t m) {
	return x * m + y;
}

inline float calcul(float *restrict val_new, float *restrict val, size_t m, size_t n) {
	// Parcours des éléments du domaine
	float error = 0.0f;
#pragma omp parallel for collapse(2) shared(val, val_new) reduction(max: error)
	for (size_t j = 1; j < n - 1; ++j) {
		for (size_t i = 1; i < m - 1; ++i) {
			// Calcul des décalages d'indices en fonction des bords
			// Calcul de la nouvelle valeur
			val_new[offset(j, i, m)] = 0.25f * (val[offset(j, i + 1, m)] + val[offset(j, i - 1, m)] + val[offset(j - 1, i, m)] + val[offset(j + 1, i, m)]);

			error = fmaxf(error, fabsf(val_new[offset(j, i, m)] - val[offset(j, i, m)]));
		}
	}
	return error;
}

inline void swap(float *restrict val, float *restrict val_new, size_t m, size_t n) {
#pragma omp parallel for collapse(2) shared(val, val_new)
	for (size_t j = 1; j < n - 1; ++j) {
		for (size_t i = 1; i < m - 1; ++i) {
			val[offset(j, i, m)] = val_new[offset(j, i, m)];
		}
	}
}

uint32_t simulation(float *restrict val_new, float *restrict val, size_t nx, size_t ny, float convergence, uint32_t nite, uint32_t out, CvMat *mat, heatPoint *restrict srcsHeat, size_t numHeat) {
	uint32_t n;
	float err = 1.0f;

	/* Boucle de résolution */
	for (n = 0; (n < nite) && (err > convergence); ++n) {
		// Calcul ces nouvelles valeurs
		err = calcul(val_new, val, nx, ny); // appel de la fonction calcul

		// Échange des valeurs.
		swap(val, val_new, nx, ny); // appel de la fonction swap

		//Les points de chaleur sont persistents
		keepHeat(val, val_new, nx, ny, srcsHeat, numHeat);

		// Sortie dans une image
		if ((n % out) == 0) {
			printf("%d %f\n", n, err);
			print_img(val, mat, nx, ny, n);
		}
		++n;
	}
	return n;
}

inline void keepHeat(float *restrict val, float *restrict val_new, size_t m, size_t n, const heatPoint *restrict srcs, size_t numHeat) {
	for (size_t i = 0; i < numHeat; ++i) {
		val[offset(srcs[i].x, srcs[i].y, m)] = 1.0f;
		val_new[offset(srcs[i].x, srcs[i].y, m)] = 1.0f;

		val[offset(srcs[i].x, srcs[i].y + 1, m)] = 1.0f;
		val_new[offset(srcs[i].x, srcs[i].y + 1, m)] = 1.0f;

		val[offset(srcs[i].x + 1, srcs[i].y, m)] = 1.0f;
		val_new[offset(srcs[i].x + 1, srcs[i].y, m)] = 1.0f;

		val[offset(srcs[i].x + 1, srcs[i].y + 1, m)] = 1.0f;
		val_new[offset(srcs[i].x + 1, srcs[i].y + 1, m)] = 1.0f;

		val[offset(srcs[i].x - 1, srcs[i].y, m)] = 1.0f;
		val_new[offset(srcs[i].x - 1, srcs[i].y, m)] = 1.0f;

		val[offset(srcs[i].x, srcs[i].y - 1, m)] = 1.0f;
		val_new[offset(srcs[i].x, srcs[i].y - 1, m)] = 1.0f;

		val[offset(srcs[i].x - 1, srcs[i].y - 1, m)] = 1.0f;
		val_new[offset(srcs[i].x - 1, srcs[i].y - 1, m)] = 1.0f;

		val[offset(srcs[i].x - 1, srcs[i].y + 1, m)] = 1.0f;
		val_new[offset(srcs[i].x - 1, srcs[i].y + 1, m)] = 1.0f;

		val[offset(srcs[i].x + 1, srcs[i].y - 1, m)] = 1.0f;
		val_new[offset(srcs[i].x + 1, srcs[i].y - 1, m)] = 1.0f;
	}
}
