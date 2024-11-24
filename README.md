# 9-Point Laplacian 2D Algorithm with DOACROSS Parallelism in OpenMP
## Project Overview
This project implements the 9-point Laplacian 2D algorithm using DOACROSS parallelism with OpenMP. The DOACROSS parallelism technique ensures efficient parallel execution while handling data dependencies across iterations.

### What is Laplacian 9-point 2D algorithm
The 9-point Laplacian stencil is a numerical method used to approximate the Laplacian operator over a 2D grid. The Laplacian algorithm is a numerical method widely used in fields such as computational physics, image processing, and numerical simulations for solving partial differential equations (PDEs) over a two-dimensional grid. For a grid point (i,j), the formula is:
```math
\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} 
```
```math
u_{i,j} = \frac{u_{i-1,j-1} + u_{i-1,j+1} + u_{i+1,j-1} + u_{i+1,j+1} + 2(u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1}) - 12u_{i,j}}{6h^2}
```

## Features
Parallelization with OpenMP: Utilizes DOACROSS dependencies to achieve efficient parallelism while preserving the correctness of computations.

Performance Metrics: Measures runtime performance with varying numbers of threads and provides data for performance analysis.

Scalability: Designed to handle large grid sizes efficiently using multi-threaded parallel execution.

Visualizing Results: Use the Python plotting script to visualize the relationship between runtime and the number of threads:

## code
main_code.cpp : Consist of four functionalities for each techniques for parallelisation .
First Enter dimension of the matrix (between 0 and 10,000) however for small values of n parallelisation is not needed since overhead associated with it is too high resulting in better sequential execution .
#### Techniques used : Guided Scheduling with or without Tiling and SIMD parallelisation .

seq.cpp : Provides input for cuda file speedups_with_cuda.cu

speedups_with_cuda.cu : We used cuda to leverage GPU processing capabilities and getting a better speedup than provided by conventional techiques .

bash_file : for running main_code.cpp and user can run speedups_with_cuda.cu file directly from this .

## Analysis
efficiency.ipynb : Contains speedup and efficiency comparison between various matrix sizes implemented using several parallelism methods like Guided Scheduling , Tiling with different block sizes and SIMD.

#### Guided Scheduling -
In guided scheduling, the chunk size decreases over time. Larger chunks are assigned initially, and as more iterations are completed, smaller chunks are given to threads. This approach can reduce overhead toward the end of the loop as fewer iterations are left.​
This scheduling is often beneficial when there’s high variation in the time required for different parts of the loop.​

Example usage: #pragma omp for schedule(guided, chunk_size)​

#### Tiling -
Tiling divides a large matrix into smaller blocks to improve parallelism and memory access. In the Laplacian solver, it enhances cache locality by processing subgrids independently, reducing cache misses and speeding up computation. However, its effectiveness varies: it works well for large problems where memory bandwidth is a bottleneck but may be less effective for smaller problems or when there's load imbalance between threads.​
In our project we have used various block sizes for analysing how would it affect execution time.

#### SIMD -
Allows multiple data elements to be processed simultaneously with a single instruction, improving parallelism and performance. In the Laplacian solver, SIMD can be used to perform multiple calculations (such as updating neighboring grid points) in parallel, effectively reducing the number of instructions and speeding up execution. By leveraging SIMD, the solver can utilize the full processing power of modern CPUs, especially in high-performance computing tasks, resulting in significant performance gains without altering the structure of the algorithm.​

These along with some inferences are available in this ipynb file.

## Simulations
This section demonstrates some of the application of the laplacian 9-point stencil like heat simulation and edge detection in images .
Edge Detection : First run the cpp file and then run the python file . 
Heat Simulation : Directly run the python file .

Contact
For questions or feedback, reach out to kartik_s@cs.iitr.ac.in.
