#include <iostream>
#include <cstdlib>
#include <chrono>    // For time measurement using chrono
#include <omp.h>

#define N 282         // Grid size (N x N)
#define MAX_ITER 1000  // Number of iterations for the algorithm

// Function to initialize the grid with random values
void initialize_grid(double grid[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i][j] = rand() ; // Random initialization
        }
    }
}

// Sequential implementation of the 9-point Laplacian
void sequential_laplacian(double input[N][N], double output[N][N]) {
    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                output[i][j] = (input[i-1][j-1] + input[i-1][j] + input[i-1][j+1] +
                                 input[i][j-1] + 4.0 * input[i][j] + input[i][j+1] +
                                 input[i+1][j-1] + input[i+1][j] + input[i+1][j+1]) / 4.0;
            }
        }
    }
}

// Parallel implementation of the 9-point Laplacian using OpenMP with DOACROSS parallelism
void parallel_laplacian(double input[N][N], double output[N][N], int num_threads) {
    // Set the number of threads dynamically
    omp_set_num_threads(num_threads);

    // #pragma omp parallel for
    for (int iter = 0; iter < MAX_ITER; iter++) {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                output[i][j] = (input[i-1][j-1] + 2.0 * input[i-1][j] + input[i-1][j+1] +
                                 2.0 * input[i][j-1] - 12.0 * input[i][j] + 2.0 * input[i][j+1] +
                                 input[i+1][j-1] + 2.0 * input[i+1][j] + input[i+1][j+1]) / 4.0;
            }
        }
    }
}

int main() {
    double input[N][N], output_seq[N][N], output_par[N][N];

    // Initialize the grid with random values
    initialize_grid(input);

    // Measure the time for the sequential execution using chrono
    auto start_time = std::chrono::high_resolution_clock::now();
    sequential_laplacian(input, output_seq);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sequential_duration = end_time - start_time;
    std::cout << "Sequential execution time: " << sequential_duration.count() << " seconds\n";

    // Test with different numbers of threads
    for (int threads = 1; threads <= 8; threads *= 2) {  // Run for 1, 2, 4, 8 threads
        std::cout << "\nRunning with " << threads << " threads...\n";

        // Measure the time for the parallel execution using OpenMP and chrono
        start_time = std::chrono::high_resolution_clock::now();
        parallel_laplacian(input, output_par, threads);
        end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> parallel_duration = end_time - start_time;
        
        std::cout << "Parallel execution time with " << threads << " threads: " 
                  << parallel_duration.count() << " seconds\n";

        // Calculate and display the speedup
        std::cout << "Speedup with " << threads << " threads: " 
                  << sequential_duration.count() / parallel_duration.count() << "\n";
    }

    return 0;
}