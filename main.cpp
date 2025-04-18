#include <iostream>
#include <random>
#include <chrono>
#include <mkl.h>
#include <omp.h>
#include <fstream>
#include <cmath>

using namespace std;

int main() {
    srand(time(NULL));
    mkl_set_num_threads(4);

    ofstream outfile("matrix_results.txt");
    if (!outfile.is_open()) {
        cerr << "Cannot open output file!" << endl;
        return 1;
    }

    double alpha = 1.0;
    double beta = 0.0;

    for (int i = 4; i <= 13; i++) {
        int N = pow(2, i);  // matrix size N x N

        double* A = (double*)mkl_malloc(N * N * sizeof(double), 64);
        double* B = (double*)mkl_malloc(N * N * sizeof(double), 64);
        double* C = (double*)mkl_malloc(N * N * sizeof(double), 64);

        for (int j = 0; j < N * N; ++j) {
            A[j] = ((double)rand() / RAND_MAX);
            B[j] = ((double)rand() / RAND_MAX);
        }

        auto start = chrono::high_resolution_clock::now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N, alpha, A, N, B, N, beta, C, N);
        auto stop = chrono::high_resolution_clock::now();

        double time_sec = chrono::duration<double>(stop - start).count();
        double flops = 2.0 * N * N * N; // 2*N^3 operations
        double gflops = flops / (time_sec * 1e9);

        double total_bytes = 3.0 * N * N * sizeof(double); // A, B, C
        double bandwidth = total_bytes / (1024 * 1024 * 1024 * time_sec); // in GB/s

        cout << "Matrix size: " << N << "x" << N << endl;
        cout << "Time: " << time_sec << " sec" << endl;
        cout << "GFLOPS: " << gflops << endl;
        cout << "Bandwidth: " << bandwidth << " GB/s" << endl;
        cout << "---------------------------------------------------" << endl;

        outfile << N << " " << gflops << " " << bandwidth << endl;

        mkl_free(A); mkl_free(B); mkl_free(C);
    }

    outfile.close();
    return 0;
}
