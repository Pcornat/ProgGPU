#ifndef PROGOMP_STRUCTURES_H
#define PROGOMP_STRUCTURES_H

#include <stdint.h>
#include <opencv2/core/core_c.h>

typedef struct src_data {
	uint32_t x; //Coordonnée x du point de chaleur
	uint32_t y; //Coordonnée y du point de chaleur
	float t; //Température
} src_t;

/**
 * Écris une image de la simulation à un instant t donné.
 * @param img Tableau de valeur pour l'image.
 * @param mat Image OpenCV.
 * @param nx Largeur de l'image.
 * @param ny Hauteur de l'image.
 * @param time Instant t donné de la simulation où l'enregistre l'image.
 */
void print_img(float *img, CvMat *mat, size_t nx, size_t ny, uint32_t time);

/**
 * Initialise la matrice de simulation.
 * @param t Matrice de simulation.
 * @param nx Largeur de la matrice.
 * @param ny Hauteur de la matrice.
 */
void init(float *t, size_t nx, size_t ny);

/**
 * Initialisation aléatoire des sources.
 * @param s Tableau des sources.
 * @param nx Largeur de la grille pour placer les sources.
 * @param ny Hauteur de la grille pour placer les sources.
 * @param n Le nombre de sources voulues.
 */
void init_src(src_t *s, size_t nx, size_t ny, size_t n);

#endif //PROGOMP_STRUCTURES_H
