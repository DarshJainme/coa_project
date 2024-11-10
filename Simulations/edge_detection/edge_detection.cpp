#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

long double kernel[3][3] = {
    {1.0, 4.0, 1.0},
    {4.0, -20.0, 4.0},
    {1.0, 4.0, 1.0}
};

// applying kernel over image to detect edges
vector<vector<long double>> apply_laplacian(const vector<vector<long double>>& pix, int wid, int hgt, int num_threads) {
    vector<vector<long double>> result(hgt, vector<long double>(wid, 0.0));
    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for (int y = 1; y < hgt - 1; y++) {
        for (int x = 1; x < wid - 1; x++) {
            
            long double sum = 0.0;

            for (int ky = -1; ky <= 1; ky++) { 
                for (int kx = -1; kx <= 1; kx++) {
                    long double val = pix[y + ky][x + kx];
                    sum += val * kernel[ky + 1][kx + 1];
                }
            }
            
            result[y][x] = min(255.0L, max(0.0L, sum + 128.0L));
        
        }
    }

    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        cerr << "Usage: edge_detection_cpp <input_file> <output_file> <hgt> <wid> <num_threads>\n";  // How to pass arguments
        return 1;
    }

    string input_file = argv[1];
    string output_file = argv[2];
    int hgt = stoi(argv[3]);
    int wid = stoi(argv[4]);
    int num_threads = stoi(argv[5]);

    vector<vector<long double>> pix(hgt, vector<long double>(wid));
    ifstream input(input_file);

    if (!input) {
        cerr << "Error: Cannot open input file.\n";
        return 1;
    }

    // Reading image pixel values from input.txt file   
    for (int y = 0; y < hgt; ++y) {
        for (int x = 0; x < wid; ++x) {
            input >> pix[y][x];
        }
    }

    input.close();

    auto start = high_resolution_clock::now();
    vector<vector<long double>> result_data = apply_laplacian(pix, wid, hgt, num_threads);
    auto end = high_resolution_clock::now();
    long double elapsed_time = duration<double>(end - start).count(); // time calculated using specified threads value

    ofstream output(output_file);
    if (!output) {
        cerr << "Error: Cannot open output file.\n";
        return 1;
    }

    for (int y = 0; y < hgt; ++y) {
        for (int x = 0; x < wid; ++x) {
            output << result_data[y][x] << " "; // writing the output back to output.txt file so it can be used by python file
        }
        output << "\n";
    }
    output.close();

    cout << elapsed_time << endl;
    return 0;
}