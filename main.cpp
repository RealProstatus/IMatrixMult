#include <iostream>
#include <random>
#include <chrono>
#include <mkl.h>
#include <omp.h>
#include <fstream>
#include <cmath>

using namespace std;


double alpha = 1.0;
double beta = 0.0;

int i = 12;

void my_dgemm(int N, double* A, double* B, double* C, double alpha, double beta) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        N, N, N, alpha, A, N, B, N, beta, C, N);
}

int main() {
    srand(time(NULL));
    mkl_set_num_threads(4);

    ofstream outfile("matrix_results.txt");
    if (!outfile.is_open()) {
        cerr << "Cannot open output file!" << endl;
        return 1;
    }

    int N = pow(2, i);  // matrix size N x N

    double* A = (double*)mkl_malloc(N * N * sizeof(double), 64);
    double* B = (double*)mkl_malloc(N * N * sizeof(double), 64);
    double* C = (double*)mkl_malloc(N * N * sizeof(double), 64);

    for (int j = 0; j < N * N; ++j) {
        A[j] = ((double)rand() / RAND_MAX);
        B[j] = ((double)rand() / RAND_MAX);
    }

    for (int loop = 0; loop <= 5; loop++) {
        my_dgemm(N, A, B, C, alpha, ::beta);
    }

    mkl_free(A); mkl_free(B); mkl_free(C);
    return 0;
}
