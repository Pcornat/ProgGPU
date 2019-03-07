#include "structures.h"
#include <stdio.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <omp.h>
#include <time.h>

//#define max(a, b) ((a>b)?a:b)
//#define min(a, b) ((a<b)?a:b)

void print_img(float *img, CvMat *mat, size_t nx, size_t ny, uint32_t time) {
	char img_name[500];
	double start, end;
	size_t sx = 1, sy = 1;

	start = omp_get_wtime();
	sprintf(img_name, "img_%05d.png", time);

	if (nx < 256 || ny < 256) {
		sx = 256 / nx;
		sy = 256 / ny;
	}

	unsigned char *mat_dat = mat->data.ptr;
	for (size_t j = 0; j < ny; ++j)
		for (size_t i = 0; i < nx; ++i) {
			uchar r, g, b;
			float t = img[i + (ny - 1 - j) * nx];

			b = (uchar) (255 * (t < 0.5f ? sin(2.0 * M_PI * t) : 0.0));
			g = (uchar) (255 * (t * t * t));
			r = (uchar) (255 * sqrtf(t));

			for (size_t jj = 0; jj < sy; ++jj)
				for (size_t ii = 0; ii < sx; ++ii) {
					mat_dat[(i * sx + ii + (j * sy + jj) * nx * sx) * 3] = b;
					mat_dat[(i * sx + ii + (j * sy + jj) * nx * sx) * 3 + 1] = g;
					mat_dat[(i * sx + ii + (j * sy + jj) * nx * sx) * 3 + 2] = r;
				}
		}
	cvSaveImage(img_name, mat, NULL);
	end = omp_get_wtime();
	printf("Écriture de l'image : %s (%f s)\n", img_name, end - start);
}

void init(float *t, size_t nx, size_t ny) {
	for (size_t j = 0; j < nx; ++j)
		for (size_t i = 0; i < nx; ++i)
			t[i + j * nx] = 0.0f;
}

void init_src(src_t *s, size_t nx, size_t ny, size_t n) {
	srand(time(NULL));
	for (size_t i = 0; i < n; i++) {
		s[i].x = (uint32_t) ((rand() / ((float) RAND_MAX)) * nx);
		s[i].y = (uint32_t) ((rand() / ((float) RAND_MAX)) * ny);
		s[i].t = rand() / ((float) RAND_MAX);
	}
}
