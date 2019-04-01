#ifndef PROGGPU_COMPUTE_FUNCTIONS_H
#define PROGGPU_COMPUTE_FUNCTIONS_H

#include <stddef.h>
#include <stdint.h>
#include <opencv2/core/core_c.h>
#include <images.h>

/**
 *
 * \param x The x coordinates
 * \param y The y coordinates
 * \param m The number of line.
 * \return
 */
#ifdef NDEBUG

extern inline
#endif
size_t offset(size_t x, size_t y, size_t m);

/**
 *
 * \param val_new Tableau contenant les nouvelles valeurs.
 * \param val Tableau contenant les anciennes valeurs.
 * \param m Nombre d'éléments en X.
 * \param n Nombre d'éléments en Y
 * \param v Coefficient.
 */
#ifdef NDEBUG
inline
#endif

float calcul(float *restrict val_new, float *restrict val, size_t m, size_t n);

/**
 * Mise à jour des sources de chaleur.
 * \param val Tableau contenant les valeurs.
 * \param m Nombre d'élément en X.
 * \param n Nombre d'élément en Y.
 * \param src Tableau contenant les sources.
 * \param n_src Nombre de sources.
 */
#ifdef NDEBUG
inline
#endif

void swap(float *restrict val, float *restrict val_new, size_t m, size_t n);

/**
 * Processus de simulation de l'équation de la chaleur en 2D
 * \param val_new Tableau contenant les nouvelles valeurs.
 * \param val Tableau contenant les anciennes valeurs.
 * \param nx Nombre d'élément en X.
 * \param ny Nombre d'élément en Y.
 * \param v Coefficient.
 * \param convergence La convergence
 * \param nite Nombre d'itération maximale.
 * \param src Tableau contenant les sources de chaleur.
 * \param nsrc Taille du tableau de sources.
 * \param out Sortie d'images toutes les « out » fois.
 * \param mat Images générées durant la simulation.
 * \return Le nombre d'itération effectuée.
 */
uint32_t simulation(float *restrict val_new, float *restrict val, size_t nx, size_t ny, float convergence, uint32_t nite, uint32_t out, CvMat *restrict mat);


#endif //PROGGPU_COMPUTE_FUNCTIONS_H
