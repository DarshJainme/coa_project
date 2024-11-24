#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <fstream>

using namespace std;

// Sequential implementation of the 9-point Laplacian
void sequential_laplacian(vector<vector<double>> &input, vector<vector<double>> &output, int N)
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

void write_execution_time(float time)
{
    ofstream outfile("seq_execution_time.txt"); // Open file in append mode
    if (outfile.is_open())
    {
        outfile << time;
        outfile.close();
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

    auto start_time = chrono::high_resolution_clock::now();
    sequential_laplacian(input, output, N);
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> sequential_duration = end_time - start_time;
    cout << "*************************************************************************\n";
    cout << "For N = " << N << endl;
    cout << "Sequential execution time is: " << sequential_duration.count() << " seconds\n\n";

    write_execution_time(sequential_duration.count());

    delete input_ptr;
    delete output_ptr;
}

int main()
{
    int N = 10000;
    driver_code(N);

    return 0;
}
