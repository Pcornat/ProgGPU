//
// Created by postaron on 27/02/2019.
//

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>


int32_t check_null(void *ptr) {
	if (ptr == NULL) {
		fprintf(stderr, "Error, NULL pointer.\n");
		return EXIT_FAILURE;
	} else
		return EXIT_SUCCESS;
}

int32_t check_perror(int val) {
	if (val != 0) {
		perror("Erreur :");
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}