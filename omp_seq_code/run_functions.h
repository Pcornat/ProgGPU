//
// Created by postaron on 26/02/2019.
//

#ifndef PROGGPU_RUN_FUNTIONS_H
#define PROGGPU_RUN_FUNTIONS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "images.h"

/**
 * Configuration des variables avec le fichier passé en paramètre.
 * @param filename Nom du fichier.
 * @param matrix Matrice pour la simulation.
 * @param newMatrix Matrice contenant les nouvelles valeurs. Nécessaire pour la simulation.
 * @param matCol Nombre de colonne de la matrice.
 * @param matRow Nombre de ligne de la matrice.
 * @param numIter Le nombre d'itération pour la simulation.
 * @param coeffD Le coefficient D pour l'équation de la chaleur (pas utilisé pour l'instant).
 * @param sortieImage La période de sortie d'image en fonction du nombre d'itération.
 * @param heatPoint Tableau des sources de chaleur.
 * @param numHeatPoint Nombre de source de chaleur.
 * @return Vrai : réussi. Faux : nettoyage assuré, échec.
 */
bool run_config(const char *filename, float **matrix, float **newMatrix, size_t *matCol, size_t *matRow, uint32_t *numIter, uint32_t *sortieImage);

/**
 * Libère la mémoire allouée dynamiquement.
 * @param matrix Matrice pour la simulation.
 * @param newMatrix Matrice contenant les nouvelles valeurs. Nécessaire pour la simulation.
 * @param heatPoint Tableau des sources de chaleur.
 */
void end_simulation(float *restrict matrix, float *restrict newMatrix);

#endif //PROGGPU_RUN_FUNTIONS_H