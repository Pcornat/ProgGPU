#ifndef PROGOMP_STRUCTURES_H
#define PROGOMP_STRUCTURES_H

#include <stdint.h>
#include <opencv2/core/core_c.h>

/**
 * Écris une image de la simulation à un instant t donné.
 * @param img Tableau de valeur pour l'image.
 * @param mat Image OpenCV.
 * @param nx Largeur de l'image.
 * @param ny Hauteur de l'image.
 * @param time Instant t donné de la simulation où l'enregistre l'image.
 */
void print_img(float *restrict img, CvMat *restrict mat, size_t nx, size_t ny, uint32_t time);

#endif //PROGOMP_STRUCTURES_H
