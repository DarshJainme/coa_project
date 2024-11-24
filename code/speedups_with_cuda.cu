// speedups_with_cuda.cu
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <vector>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

__global__ void laplacianKernel(float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        output[i * N + j] = 0.25f * (2.0 * input[(i-1) * N + j] + 2.0 * input[(i+1) * N + j] + 
                                        2.0 * input[i * N + (j-1)] + 2.0 * input[i * N + (j+1)] + 
                                        input[(i-1) * N + (j-1)] + input[(i+1) * N + (j+1)] + 
                                        input[(i-1) * N + (j+1)] + input[(i+1) * N + (j-1)] -
                                        12.0 * input[i * N + j]);
    }
}

int read_time(float& time, string file_name) {
    ifstream infile("seq_execution_time.txt");
    if (!infile.is_open())
    {
        cerr << "Error: Could not open 'execution_time.txt' for reading.\n";
        return 1;
    }

    cout << "Reading execution times from 'execution_time.txt':\n";
    string line;
    while (getline(infile, line))
    {
        time = stof(line);
    }

    infile.close();
    return 0;
}

int main() {
    // Testing CUDA speedup for 10k x 10k matrix
    int N = 10000;
    srand(time(NULL));
    float* input = new float[N * N];
    float* output = new float[N * N];

    // Initialize input with random values
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            input[i*N + j] = rand();
        }
    }

    float *d_matrix, *d_result;
    cudaMalloc((void**)&d_matrix, N * N * sizeof(float));
    cudaMalloc((void**)&d_result, N * N * sizeof(float));

    cudaMemcpy(d_matrix, input, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch kernel
    laplacianKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_result, N);

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float cuda_time;
    cudaEventElapsedTime(&cuda_time, start, stop);

    float sequential_time;
    read_time(sequential_time, "execution_time.txt");

    // Output the timing output
    cout << "Sequential execution time is: " << sequential_time << " seconds\n\n";
    cout << "Execution time with CUDA: " << cuda_time << " ms" << endl;
    cout << "Speedup with CUDA: " << sequential_time * 1000 / cuda_time << "\n\n";

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(output, d_result, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_matrix);
    cudaFree(d_result);

    delete[] input;
    delete[] output;

    return 0;
}
