#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

namespace py = pybind11;

// Forward declaration of the matrix multiply function from our CUDA file
extern int MatrixMultiply(int argc, char** argv, int block_size, 
                          const dim3& dimsA, const dim3& dimsB);

// Wrapper function that takes NumPy arrays and passes them to our CUDA implementation
py::array_t<float> matrix_multiply_cuda(py::array_t<float> A, py::array_t<float> B, int block_size = 32) {
    // Get array info and check dimensions
    py::buffer_info A_info = A.request();
    py::buffer_info B_info = B.request();
    
    if (A_info.ndim != 2 || B_info.ndim != 2)
        throw std::runtime_error("Input must be 2-D NumPy arrays");
    
    // Extract dimensions
    int A_rows = A_info.shape[0];
    int A_cols = A_info.shape[1];
    int B_rows = B_info.shape[0];
    int B_cols = B_info.shape[1];
    
    if (A_cols != B_rows)
        throw std::runtime_error("Incompatible matrix dimensions for multiplication");
    
    // Create output array
    py::array_t<float> C = py::array_t<float>({A_rows, B_cols});
    py::buffer_info C_info = C.request();
    
    // Define CUDA dimensions
    dim3 dimsA(A_cols, A_rows, 1);
    dim3 dimsB(B_cols, B_rows, 1);
    
    // Allocate host memory for matrices A and B
    float* h_A;
    float* h_B;
    float* h_C;
    
    // Allocate CUDA memory
    unsigned int size_A = A_rows * A_cols;
    unsigned int size_B = B_rows * B_cols;
    unsigned int size_C = A_rows * B_cols;
    
    unsigned int mem_size_A = sizeof(float) * size_A;
    unsigned int mem_size_B = sizeof(float) * size_B;
    unsigned int mem_size_C = sizeof(float) * size_C;
    
    // Allocate host memory
    cudaMallocHost(&h_A, mem_size_A);
    cudaMallocHost(&h_B, mem_size_B);
    cudaMallocHost(&h_C, mem_size_C);
    
    // Copy data from NumPy arrays to host memory
    float* A_ptr = static_cast<float*>(A_info.ptr);
    float* B_ptr = static_cast<float*>(B_info.ptr);
    
    for (int i = 0; i < size_A; i++) {
        h_A[i] = A_ptr[i];
    }
    
    for (int i = 0; i < size_B; i++) {
        h_B[i] = B_ptr[i];
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(reinterpret_cast<void**>(&d_A), mem_size_A);
    cudaMalloc(reinterpret_cast<void**>(&d_B), mem_size_B);
    cudaMalloc(reinterpret_cast<void**>(&d_C), mem_size_C);
    
    // Create a stream
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    
    // Copy host memory to device
    cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream);
    
    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(B_cols / threads.x, A_rows / threads.y);
    
    // Execute the kernel
    if (block_size == 16) {
        MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, A_cols, B_cols);
    } else {
        MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, A_cols, B_cols);
    }
    
    // Copy result from device to host
    cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Copy result to output NumPy array
    float* C_ptr = static_cast<float*>(C_info.ptr);
    for (int i = 0; i < size_C; i++) {
        C_ptr[i] = h_C[i];
    }
    
    // Clean up memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return C;
}

PYBIND11_MODULE(cuda_matrix_mul, m) {
    m.doc() = "CUDA Matrix Multiplication module using pybind11";
    
    m.def("matrix_multiply", &matrix_multiply_cuda, 
          "Multiply two matrices using CUDA",
          py::arg("A"), py::arg("B"), py::arg("block_size") = 32);
} 