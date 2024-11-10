import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

GRID_SIZE = 75
MAX_ITER = 500

alpha = 3
dx = 2

dt = (dx ** 2) / (4 * alpha)
gamma = (alpha * dt) / (dx ** 2)

# Initialize the grid
GRID = np.empty((MAX_ITER, GRID_SIZE, GRID_SIZE))

INIT_TEMP = 0

# Set the initial condition
GRID.fill(INIT_TEMP)

# Add random hotspots to the grid
num_hotspots = 75  
hot_temp_range = (80,100)  
pos_hotspot = [(np.random.randint(1, GRID_SIZE-1), np.random.randint(1, GRID_SIZE-1)) for _ in range(num_hotspots)]

# Assign random temperatures to the selected positions
hotspot_temp = {}
for pos in pos_hotspot:
    hotspot_temp[pos] = np.random.uniform(hot_temp_range[0], hot_temp_range[1])
    GRID[0, pos[0], pos[1]] = hotspot_temp[pos]

# Calculate the next time step
def calculate(GRID):
    for k in range(0, MAX_ITER - 1, 1):
        for i in range(1, GRID_SIZE - 1, 1):
            for j in range(1, GRID_SIZE - 1, 1):
                if (i, j) in hotspot_temp:
                    GRID[k + 1, i, j] = hotspot_temp[(i, j)]
                else:
                    GRID[k + 1, i, j] = gamma * (0.5*GRID[k][i - 1][j - 1] + 0.5*GRID[k][i - 1][j + 1] + 0.5*GRID[k][i + 1][j - 1] + 0.5*GRID[k][i + 1][j + 1] + GRID[k][i + 1][j] + GRID[k][i - 1][j] + GRID[k][i][j + 1] + GRID[k][i][j - 1] - 6 * GRID[k][i][j]) + GRID[k][i][j]

    return GRID

# Function to plot the heatmap at time step k
def plotheatmap(GRID_K, k):
    plt.clf()
    plt.title(f"Temperature at t = {k * dt:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pcolormesh(GRID_K, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

GRID = calculate(GRID)

# Animation
def animate(k):
    plotheatmap(GRID[k], k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=MAX_ITER, repeat=False)
anim.save("heat_simulation.gif")

print("Simulation completed")