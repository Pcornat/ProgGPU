#include "compute_functions.h"

inline void calcul(float *val_new, float *val, size_t nx, size_t ny, float v) {
	// Parcours des éléments du domaine
#pragma omp parallel for collapse(2) shared(val, val_new)
	for (size_t j = 0; j < ny; ++j) {
		for (size_t i = 0; i < nx; ++i) {
			// Calcul des décalages d'indices en fonction des bords
			size_t i_g, i_d, j_h, j_b;
			i_g = (i == 0) ? 0 : i - 1;
			j_b = (j == 0) ? 0 : j - 1;
			i_d = (i == nx - 1) ? nx - 1 : i + 1;
			j_h = (j == ny - 1) ? ny - 1 : j + 1;

			// Calcul de la nouvelle valeur
			val_new[i + j * nx] = val[i + j * nx] * (1.0f - 4.0f * v) + v * (val[i_g + j * nx] + val[i_d + j * nx] + val[i + j_h * nx] + val[i + j_b * nx]);
		}
	}
}

inline void update_src(float *val, size_t nx, size_t ny, src_t *src, size_t n_src) {
	for (size_t i = 0; i < n_src; i++) {
		val[src[i].x + src[i].y * nx] = src[i].t;
	}
}

/**
 * Calcul de la norme L2 de la différence des valeurs
 * @param val_new Tableau contenant les nouvelles valeurs calculées
 * @param val Tableau contenant les anciennes valeurs
 * @param n Nombre total d'éléments dans le domaine de calcul
 * @return
 */
inline float reduction(float *val_new, float *val, size_t n) {
	float err = 0.0;
	// Réduction des n éléments
	// La norme L2 est la somme des carrés de la différence entre val_new et val.
	for (size_t i = 0; i < n; i++) {
		err += (val_new[i] - val[i]) * (val_new[i] - val[i]);
	}
	return err;
}

int simulation(float *val_new, float *val, size_t nx, size_t ny, float v, float convergence, uint32_t nite, src_t *src, size_t nsrc, uint32_t out, CvMat *mat) {
	uint32_t n;
	float err = 10000, *tmp = NULL;

	/* Boucle de résolution */
	for (n = 1; (n < nite) && (err > convergence); ++n) {
		// Calcul ces nouvelles valeurs
		calcul(val_new, val, nx, ny, v); // appel de la fonction calcul

		// Mise à jour des sources
		update_src(val_new, nx, ny, src, nsrc); // appel de la fonction update_src

		// Calcul de la norme du résidu
		err = reduction(val_new, val, nx * ny); // appel de la fonction reduction

		// Echange des tableaux de valeur
		tmp = val_new;  // Utiliser le pointeur 'tmp' pour échanger les valeurs des pointeurs val_new et val
		val_new = val;
		val = tmp;
		// Sortie dans une image
		if (n % out == 0) {
			printf("%d %f\n", n, err);
			print_img(val, mat, nx, ny, n);
		}
		++n;
	}
	return n;
}