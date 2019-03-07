#ifndef PROGGPU_COMPUTE_FUNCTIONS_H
#define PROGGPU_COMPUTE_FUNCTIONS_H

#include <stddef.h>
#include <stdint.h>
#include <opencv2/core/core_c.h>
#include <structures.h>

/**
 *
 * @param val_new Tableau contenant les nouvelles valeurs.
 * @param val Tableau contenant les anciennes valeurs.
 * @param nx Nombre d'éléments en X.
 * @param ny Nombre d'éléments en Y
 * @param v Coefficient.
 */
inline void calcul(float *val_new, float *val, size_t nx, size_t ny, float v);

/**
 * Mise à jour des sources de chaleur.
 * @param val Tableau contenant les valeurs.
 * @param nx Nombre d'élément en X.
 * @param ny Nombre d'élément en Y.
 * @param src Tableau contenant les sources.
 * @param n_src Nombre de sources.
 */
inline void update_src(float *val, size_t nx, size_t ny, src_t *src, size_t n_src);

/**
 * Processus de simulation de l'équation de la chaleur en 2D
 * @param val_new Tableau contenant les nouvelles valeurs.
 * @param val Tableau contenant les anciennes valeurs.
 * @param nx Nombre d'élément en X.
 * @param ny Nombre d'élément en Y.
 * @param v Coefficient.
 * @param convergence La convergence
 * @param nite Nombre d'itération maximale.
 * @param src Tableau contenant les sources de chaleur.
 * @param nsrc Taille du tableau de sources.
 * @param out Sortie d'images toutes les « out » fois.
 * @param mat Images générées durant la simulation.
 * @return Le nombre d'itération effectuée.
 */
int simulation(float *val_new, float *val, size_t nx, size_t ny, float v, float convergence, uint32_t nite, src_t *src, size_t nsrc, uint32_t out, CvMat *mat);


#endif //PROGGPU_COMPUTE_FUNCTIONS_H
