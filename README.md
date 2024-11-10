9-Point Laplacian 2D Algorithm with DOACROSS Parallelism
Project Overview
This project provides a C program that implements a 2D 9-point Laplacian stencil using DOACROSS parallelism with OpenMP. The 9-point Laplacian stencil is used in various scientific and engineering applications, such as image processing, heat distribution simulations, and solving partial differential equations (PDEs). 
The goal of this project is to leverage DOACROSS parallelism to improve computational performance by effectively managing dependencies across grid points in a parallel environment.

Table of Contents
Project Overview
Background
Features
Getting Started
Usage
Parallelism Details
Performance
References
Background
The 9-point Laplacian stencil operates on a 2D grid, updating each cell based on its neighboring values. Unlike the simpler 5-point stencil, the 9-point Laplacian also considers the diagonal neighbors, resulting in a more accurate approximation of spatial derivatives.

DOACROSS parallelism in OpenMP allows for parallelization across loop iterations with specific dependencies. Here, we apply DOACROSS parallelism to efficiently calculate each grid point in parallel while respecting the dependencies required by the Laplacian stencil.

Features
9-Point Laplacian Algorithm: Implements the 9-point Laplacian stencil for 2D grids.
DOACROSS Parallelism: Utilizes OpenMP to parallelize the calculation, with handling of dependencies across grid points.
Configurable Grid Size and Iterations: Easily adjust grid dimensions and iteration count.
Performance Timing: Measures and outputs execution time for performance analysis.
Getting Started
Prerequisites
To compile and run this project, you will need:

C Compiler with OpenMP support (e.g., GCC with -fopenmp flag).
Make (optional) if using the provided Makefile.
Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/your-repo/9-point-laplacian-doacross.git
cd 9-point-laplacian-doacross
Compile the Program:

Using the Makefile:
bash
Copy code
make
Or manually:
bash
Copy code
gcc -fopenmp -o laplacian_9pt main.c
Usage
To run the compiled program:

bash
Copy code
./laplacian_9pt [grid_size] [iterations] [num_threads]
Arguments
grid_size: Dimension of the grid (e.g., 100 for a 100x100 grid).
iterations: Number of iterations for the Laplacian calculation.
num_threads: Number of OpenMP threads to use.
Example
bash
Copy code
./laplacian_9pt 100 1000 8
This command runs the program on a 100x100 grid for 1000 iterations with 8 threads.

Parallelism Details
DOACROSS Parallelism
DOACROSS parallelism in OpenMP is implemented to manage dependencies across the grid. Each grid point update depends on its neighbors, and the DOACROSS approach ensures that necessary dependencies are respected. In particular:

Row-wise Dependency: Each cell depends on values from the previous iteration of both row and column neighbors.
DOACROSS Approach: Ensures that row dependencies are handled, enabling partial overlap across rows and allowing for efficient parallelism.
OpenMP Implementation
In the code, #pragma omp parallel for is used to enable DOACROSS parallelism. The specific dependencies in the grid are managed within the OpenMP loop to maintain the accuracy of the 9-point stencil while maximizing performance.

Performance
The program measures and outputs the total execution time for the Laplacian calculations, allowing users to analyze the effects of parallelism and grid size on performance. Timing data can be exported and analyzed to evaluate speedup as the number of threads increases.

Analyzing Performance
To analyze performance, experiment with different grid sizes, iteration counts, and thread numbers. Compare the time taken for each configuration to assess the efficiency and scalability of DOACROSS parallelism with the 9-point stencil.

References
OpenMP Documentation: https://www.openmp.org/
Laplacian Operator in Numerical Computing: Research on the 9-point stencil and its applications.
Parallel Computing with OpenMP: Study materials on DOACROSS parallelism.
