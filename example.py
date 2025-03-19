import numpy as np
import time
import cuda_matrix_mul  # Import our CUDA module

def test_matrix_multiply():
    # Create random matrices
    A = np.random.rand(1024, 1024).astype(np.float32)
    B = np.random.rand(1024, 1024).astype(np.float32)
    
    # Numpy matrix multiplication (CPU)
    start_time = time.time()
    C_numpy = np.matmul(A, B)
    numpy_time = time.time() - start_time
    print(f"NumPy multiplication time: {numpy_time:.6f} seconds")
    
    # CUDA matrix multiplication (GPU)
    start_time = time.time()
    C_cuda = cuda_matrix_mul.matrix_multiply(A, B)
    cuda_time = time.time() - start_time
    print(f"CUDA multiplication time: {cuda_time:.6f} seconds")
    
    # Verify results
    if np.allclose(C_numpy, C_cuda, rtol=1e-5, atol=1e-5):
        print("Results match!")
    else:
        print("Results don't match!")
    
    # Calculate speedup
    speedup = numpy_time / cuda_time
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    print("Testing CUDA Matrix Multiplication vs NumPy")
    test_matrix_multiply() 