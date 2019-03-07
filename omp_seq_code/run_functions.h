//
// Created by postaron on 26/02/2019.
//

#ifndef PROGGPU_RUN_FUNTIONS_H
#define PROGGPU_RUN_FUNTIONS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "structures.h"

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
bool run_config(const char *filename, float *matrix, float *newMatrix, size_t *matCol, size_t *matRow, size_t *numIter, float *coeffD, uint32_t *sortieImage, src_t *heatPoint, size_t *numHeatPoint);

/**
 * Lance la simulation avec OpenMP
 * @param matrix Matrice pour la simulation.
 * @param newMatrix Matrice contenant les nouvelles valeurs. Nécessaire pour la simulation.
 * @param matCol Nombre de colonne de la matrice.
 * @param matRow Nombre de ligne de la matrice.
 * @param numIter Le nombre d'itération pour la simulation.
 * @param coeffD Le coefficient D pour l'équation de la chaleur (pas utilisé pour l'instant).
 * @param sortieImage La période de sortie d'image en fonction du nombre d'itération.
 * @param heatPoint Tableau des sources de chaleur.
 * @param numHeatPoint Nombre de source de chaleur.
 * @param convergence Valeur à partir de laquelle on arrête la simulation.
 * @param img Image créée à partir de la simulation.
 * @return Le nombre d'itération totale effectuée.
 */
int32_t
run_openmp(float *matrix, float *newMatrix, size_t matCol, size_t matRow, size_t numIter, const float coeffD, uint32_t sortieImage, src_t *heatPoint, size_t numHeatPoint, const float convergence,
		   CvMat *img);

/**
 * Lance la simulation séquentielle.
 * @param matrix Matrice pour la simulation.
 * @param newMatrix Matrice contenant les nouvelles valeurs. Nécessaire pour la simulation.
 * @param matCol Nombre de colonne de la matrice.
 * @param matRow Nombre de ligne de la matrice.
 * @param numIter Le nombre d'itération pour la simulation.
 * @param coeffD Le coefficient D pour l'équation de la chaleur (pas utilisé pour l'instant).
 * @param sortieImage La période de sortie d'image en fonction du nombre d'itération.
 * @param heatPoint Tableau des sources de chaleur.
 * @param numHeatPoint Nombre de source de chaleur.
 * @param convergence Valeur à partir de laquelle on arrête la simulation.
 * @param img Image créée à partir de la simulation.
 * @return Le nombre d'itération totale effectuée.
 */
int32_t
run_sequentiel(float *matrix, float *newMatrix, size_t matCol, size_t matRow, size_t numIter, const float coeffD, uint32_t sortieImage, src_t *heatPoint, size_t numHeatPoint, const float convergence,
			   CvMat *img);

/**
 * Libère la mémoire allouée dynamiquement.
 * @param matrix Matrice pour la simulation.
 * @param newMatrix Matrice contenant les nouvelles valeurs. Nécessaire pour la simulation.
 * @param heatPoint Tableau des sources de chaleur.
 */
void end_simulation(float *matrix, float *newMatrix, src_t *heatPoint);

#endif //PROGGPU_RUN_FUNTIONS_H