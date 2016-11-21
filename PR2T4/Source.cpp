#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <mkl.h>

void proceso2(int N, int NTEST);
int main(int argc, char *argv[]) {
	int NTESTS = 100;
	int Ns[11] = { 128,256,512,1024,2048,3072,4096,5120,6144,7168,8192 };
	for (int i = 0; i < 11; i++)
	{
		printf("Proceso de %d test para tamano %d\n", NTESTS, Ns[i]);
		proceso2(Ns[i], NTESTS);
		printf("\n\n");
	}
	std::getchar();
	return 0;
}

void proceso2(int N, int NTEST) {
	double fin, inicio = dsecnd();
	double *A = (double*)mkl_malloc(N*N * sizeof(double), 64);
	if (A == (double*)NULL) {
		perror("Error Malloc");
		exit(1);
	}
	double *B = (double*)mkl_malloc(N*N * sizeof(double), 64);
	if (B == (double*)NULL) {
		perror("Error Malloc");
		exit(1);
	}
	double *C = (double*)mkl_malloc(N*N * sizeof(double), 64);
	if (C == (double*)NULL) {
		perror("Error Malloc");
		exit(1);
	}

	srand((unsigned int)time(NULL));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[i*N + j] = (double)rand() / (double)RAND_MAX;
			B[i*N + j] = (double)rand() / (double)RAND_MAX;
		}
	}
	inicio = dsecnd();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
	for (int i = 0; i < NTEST; i++) {
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
	}
	fin = dsecnd();
	double tiempo = (fin - inicio) / (double)NTEST;
	printf("Tiempo: %lf msec\n", tiempo*1.0e3);
	printf("GFlops: %lf\n", 2.0*pow((double)N, 3.0)*1.0e-9 / tiempo);
	mkl_free(A);
	mkl_free(B);
	mkl_free(C);
}