#include<iostream>
#include<random>
#include<chrono>
#include<mkl.h>
#include<omp.h>
#include<stdlib.h>
#include<time.h>
#include<cmath>

using namespace std;
//___________________________________________________________________________________________________________
int main() {
	srand(time(NULL));

	//mkl_set_num_threads(MKL_Get_Max_Threads());
	mkl_set_num_threads(1);

	double* A, * B, * C;
	double alpha = 1.0;
	double betta = 0.0;

	for (int i = 4; i <= 13; i++) {
		int block_size = pow(2, i);

		A = (double*)mkl_malloc(block_size * block_size * sizeof(double),64);
		B = (double*)mkl_malloc(block_size * block_size * sizeof(double), 64);
		C = (double*)mkl_malloc(block_size * block_size * sizeof(double), 64);

		for (int n = 0; n < block_size; n++) {
			for (int m = 0; m < block_size; m++) {
				double f = 1000 * ((double)rand() / RAND_MAX);
				A[n * block_size + m] = f;
				f = 1000 * ((double)rand() / RAND_MAX);
				B[n * block_size + m] = f;
			}
		}

		//auto start = omp_get_wtime();
		auto start = chrono::high_resolution_clock::now();
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, block_size, block_size, block_size, alpha, A, block_size, B, block_size, betta, C, block_size);
		auto stop = chrono::high_resolution_clock::now();
		auto res = chrono::duration_cast<chrono::milliseconds>(stop - start);
		//auto stop = omp_get_wtime();

		//auto res = stop - start;
		cout << "matrix_size = " << block_size << endl;
		cout << res.count() << " sec" << endl;
		cout << (2 * pow(block_size, 3)) / (res.count() * 1.0e6) << " GFLOPS" << endl;
		cout << "__________________________________________________________" << endl;

		mkl_free(A); mkl_free(B); mkl_free(C);
	}
	
	return 0;
}