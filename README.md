# 9-Point Laplacian 2D Algorithm with DOACROSS Parallelism in OpenMP
## Project Overview
This project implements the 9-point Laplacian 2D algorithm using DOACROSS parallelism with OpenMP. The DOACROSS parallelism technique ensures efficient parallel execution while handling data dependencies across iterations.

### What is Laplacian 9-point 2D algorithm
The 9-point Laplacian stencil is a numerical method used to approximate the Laplacian operator over a 2D grid. The Laplacian algorithm is a numerical method widely used in fields such as computational physics, image processing, and numerical simulations for solving partial differential equations (PDEs) over a two-dimensional grid. For a grid point (i,j), the formula is:
```math
\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} 
```
```math
u_{i,j} = \frac{-u_{i-1,j-1} - u_{i-1,j+1} - u_{i+1,j-1} - u_{i+1,j+1} + 4(u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1}) - 20u_{i,j}}{6h^2}
```

Features
Laplacian Kernel: Implements the 9-point stencil operator to compute second-order derivatives in a 2D grid.
Parallelization with OpenMP: Utilizes DOACROSS dependencies to achieve efficient parallelism while preserving the correctness of computations.
Performance Metrics: Measures runtime performance with varying numbers of threads and provides data for performance analysis.
Scalability: Designed to handle large grid sizes efficiently using multi-threaded parallel execution.
Directory Structure
bash
Copy code
├── src/
│   ├── main.cpp       # Contains the main implementation of the Laplacian algorithm
│   ├── utils.cpp      # Utility functions for grid initialization and performance timing
│   └── utils.h        # Header file for utility functions
├── data/
│   └── results/       # Stores timing results for different thread counts
├── plots/
│   └── plot.py        # Python script for visualizing timing results
├── Makefile           # Build automation script
└── README.md          # Project documentation
Installation and Requirements
Dependencies:

Compiler: GCC (with OpenMP support) or compatible.
Python (for plotting): matplotlib, numpy.
Setup:

Clone the repository:

bash
Copy code
git clone https://github.com/your-repo/9-point-laplacian-doacross.git
cd 9-point-laplacian-doacross
Compile the program:

bash
Copy code
make
Run the program:

bash
Copy code
./laplacian
Usage
Command-Line Options:
The program accepts parameters for grid size and number of threads. For example:

bash
Copy code
./laplacian <grid_size> <num_threads>
Example:

bash
Copy code
./laplacian 1000 8
Output:
The program outputs:

Final grid state (if requested).
Runtime for computation.
Timing results saved to the data/results/ directory.
Visualizing Results:
Use the Python plotting script to visualize the relationship between runtime and the number of threads:

bash
Copy code
python3 plots/plot.py
How It Works
Grid Initialization:
A 2D grid of user-defined size is initialized with boundary values.

Laplacian Calculation:
The Laplacian stencil computes values at each grid point based on its neighbors.

DOACROSS Parallelism:
OpenMP's DOACROSS technique is used to manage dependencies across grid rows, ensuring correctness while maximizing parallel performance.

Performance Evaluation:
The program measures and outputs timing for varying numbers of threads.

Performance Analysis
Timing data is logged for analysis.
Use the Python script to generate plots of execution time vs. thread count to evaluate scalability and parallel efficiency.
Future Enhancements
Implement additional boundary condition support (e.g., Neumann or Robin).
Extend to 3D Laplacian for volumetric simulations.
Explore other parallelism models such as MPI or hybrid MPI+OpenMP.
Contributing
Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

## Simulations
This section demonstrates some of the application of the laplacian 9-point stencil like heat simulation and edge detection in images .


License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or feedback, reach out to your-email@example.com.
