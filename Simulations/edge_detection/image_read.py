import cv2
import numpy as np
import subprocess
import matplotlib.pyplot as plt

def compile_cpp():
    result = subprocess.run(
        ["g++", "edge_detection.cpp", "-o", "edge_detection", "-fopenmp"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr)
        return False
    
    return True

def save_matrix_to_file(matrix, filename):
    np.savetxt(filename, matrix.astype(np.float64), fmt='%.18e')

def load_matrix_from_file(filename, shape):
    matrix = np.loadtxt(filename, dtype=np.float64)
    
    return matrix.reshape(shape)

def run_edge_detection(i, height, width):
    result = subprocess.run(
        ["./edge_detection", "input.txt", "output.txt", str(height), str(width), str(i)],
        capture_output=True,
        text=True
    )
    
    return float(result.stdout.strip())  # Retrieve elapsed time from stdout

def main():
    # Compiling the C++ code with OpenMP support
    if not compile_cpp():
        return
    
    # Read image as grayscale and save to input file
    image = cv2.imread("husky.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64) 
    if image is None:
        print("Failed to read the image.")
        return
    
    height, width = image.shape
    save_matrix_to_file(image, "input.txt")

    # Thread counts to be used and initialize a list for time elapsed
    threads = [1, 2, 4, 8, 16]
    times = []

    # Run edge detection for each thread count and record execution time
    for i in threads:
        elapsed_time = run_edge_detection(i, height, width)
        times.append(elapsed_time)
        print(f"Threads: {i}, Time: {elapsed_time:.4f} seconds")

    # Plot time elapsed vs. number of threads
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(threads, times, marker='o')
    plt.xlabel("Number of Threads")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time vs. Number of Threads")
    plt.grid()

    # Load the processed image and display it
    result_image = load_matrix_from_file("output.txt", (height, width)).astype(np.uint8)

    plt.subplot(1, 2, 2)
    plt.imshow(result_image, cmap='gray')
    plt.title("Edge Detection Result")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
