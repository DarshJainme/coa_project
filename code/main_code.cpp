// main code
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>
#include <vector>
#include <fstream>
#include <immintrin.h>

using namespace std;

// Sequential implementation of the 9-point Laplacian
void sequential_laplacian(vector<vector<double>> &input, vector<vector<double>> &output, int num_iter, int N)
{
    for (int iter = 0; iter < num_iter; iter++)
    {
        for (int i = 1; i < N - 1; i++)
        {
            for (int j = 1; j < N - 1; j++)
            {
                output[i][j] = (input[i - 1][j - 1] + 2.0 * input[i - 1][j] + input[i - 1][j + 1] +
                                2.0 * input[i][j - 1] - 12.0 * input[i][j] + 2.0 * input[i][j + 1] +
                                input[i + 1][j - 1] + 2.0 * input[i + 1][j] + input[i + 1][j + 1]) /
                               4.0;
            }
        }
    }
}

// Parallel implementation of the 9-point Laplacian using OpenMP
void parallel_laplacian(vector<vector<double>> &input, vector<vector<double>> &output, int num_threads, int num_iter, int N)
{
    // Set the number of threads dynamically
    omp_set_num_threads(num_threads);

    for (int iter = 0; iter < num_iter; iter++)
    {
#pragma omp parallel for collapse(2) schedule(guided)
        for (int i = 1; i < N - 1; i++)
        {
            for (int j = 1; j < N - 1; j++)
            {
                output[i][j] = (input[i - 1][j - 1] + 2.0 * input[i - 1][j] + input[i - 1][j + 1] +
                                2.0 * input[i][j - 1] - 12.0 * input[i][j] + 2.0 * input[i][j + 1] +
                                input[i + 1][j - 1] + 2.0 * input[i + 1][j] + input[i + 1][j + 1]) /
                               4.0;
            }
        }
    }
}

void laplacian_9pt_tiling(vector<vector<double>> &input, vector<vector<double>> &output, int num_threads, int num_iter, int N)
{
    int BLOCK_SIZE = 16;

    cout << "Running Tiling with Block Size " << BLOCK_SIZE << "...\n";
    omp_set_num_threads(num_threads);

    for (int iter = 0; iter < num_iter; iter++)
    {
#pragma omp parallel for collapse(2) schedule(guided)
        for (int i_block = 1; i_block < N - 1; i_block += BLOCK_SIZE)
        {
            for (int j_block = 1; j_block < N - 1; j_block += BLOCK_SIZE)
            {
                for (int i = i_block; i < min(i_block + BLOCK_SIZE, N - 1); i++)
                {
                    for (int j = j_block; j < min(j_block + BLOCK_SIZE, N - 1); j++)
                    {
                        output[i][j] = (input[i - 1][j - 1] + 2.0 * input[i - 1][j] + input[i - 1][j + 1] +
                                        2.0 * input[i][j - 1] - 12.0 * input[i][j] + 2.0 * input[i][j + 1] +
                                        input[i + 1][j - 1] + 2.0 * input[i + 1][j] + input[i + 1][j + 1]) /
                                       4.0;
                    }
                }
            }
        }
    }
}

void simd_laplacian(const vector<vector<double>> &input, vector<vector<double>> &output, int threads, int num_iter, int N)
{
    omp_set_num_threads(threads);
    for (int iter = 0; iter < num_iter; iter++)
    {
#pragma omp parallel for collapse(2) schedule(guided)
        for (int i = 1; i < N - 1; i++)
        {
            for (int j = 1; j < N - 1; j += 8)
            {
                // Process 8 elements with AVX2
                // Load 8 doubles from neighboring cells using AVX
                __m256d center = _mm256_loadu_pd(&input[i][j]);
                __m256d left = _mm256_loadu_pd(&input[i][j - 1]);
                __m256d right = _mm256_loadu_pd(&input[i][j + 1]);
                __m256d top = _mm256_loadu_pd(&input[i - 1][j]);
                __m256d bottom = _mm256_loadu_pd(&input[i + 1][j]);
                __m256d top_left = _mm256_loadu_pd(&input[i - 1][j - 1]);
                __m256d top_right = _mm256_loadu_pd(&input[i - 1][j + 1]);
                __m256d bottom_left = _mm256_loadu_pd(&input[i + 1][j - 1]);
                __m256d bottom_right = _mm256_loadu_pd(&input[i + 1][j + 1]);
                // Compute the Laplacian using weights
                __m256d result = _mm256_mul_pd(center, _mm256_set1_pd(-12.0));
                result = _mm256_add_pd(result, _mm256_mul_pd(left, _mm256_set1_pd(2.0)));
                result = _mm256_add_pd(result, _mm256_mul_pd(right, _mm256_set1_pd(2.0)));
                result = _mm256_add_pd(result, _mm256_mul_pd(top, _mm256_set1_pd(2.0)));
                result = _mm256_add_pd(result, _mm256_mul_pd(bottom, _mm256_set1_pd(2.0)));
                result = _mm256_add_pd(result, top_left);
                result = _mm256_add_pd(result, top_right);
                result = _mm256_add_pd(result, bottom_left);
                result = _mm256_add_pd(result, bottom_right);
                // Store the result back
                _mm256_storeu_pd(&output[i][j], _mm256_div_pd(result, _mm256_set1_pd(4.0)));
            }
        }
    }
}

void driver_code(int N)
{
    vector<vector<double>> *input_ptr = new vector<vector<double>>(N, vector<double>(N));
    vector<vector<double>> *output_ptr = new vector<vector<double>>(N, vector<double>(N));

    vector<vector<double>> &input = *input_ptr;
    vector<vector<double>> &output = *output_ptr;

    srand(time(NULL));
    const int threads = 16;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            input[i][j] = rand();
        }
    }

    cout<<"Running sequential execution...\n";
    auto start_time = chrono::high_resolution_clock::now();
    sequential_laplacian(input, output, 1e8 / (N * N), N);
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> sequential_duration = end_time - start_time;
    cout << "Sequential execution time is: " << sequential_duration.count() << " seconds\n\n";

    if (N >= 50)
    {
        cout << "\nRunning parallel execution with " << threads << " threads... (wITHOUT TILING)\n";

        start_time = chrono::high_resolution_clock::now();
        parallel_laplacian(input, output, threads, 1e8 / (N * N), N);
        end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> parallel_duration = end_time - start_time;

        cout << "Parallel execution time with " << threads << " threads without tiling: " << parallel_duration.count() << " seconds\n";

        cout << "Speedup with " << threads << " threads without tiling: " << sequential_duration.count() / parallel_duration.count() << "\n\n";

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        cout << "\nRunning parallel execution with " << threads << " threads...\n (WITH TILING)\n";
        
        start_time = chrono::high_resolution_clock::now();
        laplacian_9pt_tiling(input, output, threads, 1e8 / (N * N), N);
        end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> tiling_duration = end_time - start_time;

        cout << "Tiling execution time with " << threads << " threads: " << tiling_duration.count() << " seconds\n";

        cout << "Speedup with " << threads << " threads with tiling: " << sequential_duration.count() / tiling_duration.count() << "\n";

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        cout << "\nRunning parallel execution with " << threads << " threads...\n (WITH SIMD PARALLELISM)\n";
        start_time = chrono::high_resolution_clock::now();
        simd_laplacian(input, output, threads, 1e8 / (N * N), N);
        end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> simd_duration = end_time - start_time;

        cout << "SIMD execution time with " << threads << " threads: " << simd_duration.count() << " seconds\n";

        cout << "Speedup with " << threads << " threads with SIMD: " << sequential_duration.count() / simd_duration.count() << "\n";
    }
    else
    {
        cout << "Parallelization is not needed when N is small since the matrix is too small to be parallelized for laplacian calculations.\n";
        cout << "The overhead associated with parallelization is too high for small matrices and thus sequential execution is faster.\n";
    }

    delete input_ptr;
    delete output_ptr;
}

int main()
{
    cout << "Welcome to COA Project by Team Plumbers!\n";
    int N;
    cout << "Enter dimension of matrix (between 0 and 10000): ";
    cin >> N;
    if (N < 0 || N > 10000)
    {
        cout << "Error: Invalid N value. N should be positive and less than 10000.\n";
    }
    else
        driver_code(N);

    return 0;
}
