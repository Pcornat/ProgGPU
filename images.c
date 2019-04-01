#include "images.h"
#include <stdio.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <compute_functions.h>

//#define max(a, b) ((a>b)?a:b)
//#define min(a, b) ((a<b)?a:b)

void print_img(float *restrict img, CvMat *restrict mat, size_t nx, size_t ny, uint32_t time) {
	char img_name[500];
	double start, end;
	size_t sx = 1, sy = 1;

	start = omp_get_wtime();
	memset(img_name, '\0', sizeof(img_name));
	sprintf(img_name, "images/img_%05d.png", time);

	if (nx < 256 || ny < 256) {
		sx = 256 / nx;
		sy = 256 / ny;
	}

	unsigned char *restrict mat_dat = mat->data.ptr;
	for (size_t j = 0; j < ny; ++j)
		for (size_t i = 0; i < nx; ++i) {
			uchar r, g, b;
			float t = img[offset(j, i, ny)];

			b = (uchar) (255 * (t < 0.5f ? 1 : 0.0f));
			g = (uchar) (255 * (t == 1.0f ? 0 : t * 1.3f));
			r = (uchar) (255 * t);

			for (size_t jj = 0; jj < sy; ++jj)
				for (size_t ii = 0; ii < sx; ++ii) {
					mat_dat[(i * sx + ii + (j * sy + jj) * nx * sx) * 3] = b;
					mat_dat[(i * sx + ii + (j * sy + jj) * nx * sx) * 3 + 1] = g;
					mat_dat[(i * sx + ii + (j * sy + jj) * nx * sx) * 3 + 2] = r;
				}
		}
	cvSaveImage(img_name, mat, NULL); //Bug, à changer. Solution possible : compilé avec g++ et utiliser le C++
	end = omp_get_wtime();
	printf("Écriture de l'image : %s (%f s)\n", img_name, end - start);
}