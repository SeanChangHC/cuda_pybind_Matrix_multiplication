#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* Configuration Constants
 * These define the block sizes and other parameters for the CUDA kernel execution
 */
#define BLOCK_SIZE_16 16      /* Small block size option for matrix multiplication */
#define BLOCK_SIZE_32 32      /* Larger block size option for matrix multiplication */
#define N (2048*2048)         /* Total matrix size */
#define THREADS_PER_BLOCK 128 /* Number of threads per block */
#define NUM_STREAMS 4         /* Number of concurrent CUDA streams */

/* CUDA Kernel for Matrix Multiplication
 * Implements tiled matrix multiplication using shared memory for better performance
 */
template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float* C, float* A, float* B, int wA, int wB) {
	/* Calculate 2D block indices from the block ID in the grid */
	int bx = blockIdx.x;
	int by = blockIdx.y;

	/* Calculate 2D thread indices within each block */
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	/* Calculate starting index of the first sub-matrix of A for this block */
	int aBegin = wA * BLOCK_SIZE * by;

	/* Calculate ending index of sub-matrices of A for this block */
	int aEnd = aBegin + wA - 1;

	/* Calculate step size for moving through sub-matrices of A */
	int aStep = BLOCK_SIZE;

	/* Calculate starting index of the first sub-matrix of B for this block */
	int bBegin = BLOCK_SIZE * bx;

	/* Calculate step size for moving through sub-matrices of B */
	int bStep = BLOCK_SIZE * wB;

	/* Initialize accumulator for the computed element in output matrix C
	 * Each thread computes one element of the result matrix
	 */
	float Csub = 0;

	/* Main loop: process all sub-matrices of A and B required to compute this block of C
	 * This implements the tiled matrix multiplication algorithm
	 */
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		/* Allocate shared memory for sub-matrix of A
		 * This allows faster access by all threads in the block
		 */
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		/* Allocate shared memory for sub-matrix of B
		 * This reduces global memory access latency
		 */
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		/* Load data from global memory to shared memory
		 * Each thread loads one element of each matrix
		 */
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		/* Ensure all threads have loaded their data before computation begins */
		__syncthreads();

		/* Multiply sub-matrices: each thread computes one dot product
		 * Use pragma unroll to optimize inner loop performance
		 */
#pragma unroll

		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub += As[ty][k] * Bs[k][tx];
		}

		/* Ensure all computations are complete before loading new sub-matrices
		 * This synchronization is crucial for correctness
		 */
		__syncthreads();
	}

	/* Write the computed element to the output matrix C in global memory
	 * Each thread writes exactly one element
	 */
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

/* Helper Function: Initialize Matrix with a Constant Value
 * Used to set up input matrices before computation
 */
void ConstantInit(float* data, int size, float val) {
	for (int i = 0; i < size; ++i) {
		data[i] = val;
	}
}

/* CPU Implementation of Matrix Multiplication
 * Used as a reference for correctness verification and performance comparison
 */
void MatrixMulCPU(float* C, const float* A, const float* B, int wA, int hA, int wB) {
	for (int i = 0; i < hA; ++i) {
		for (int j = 0; j < wB; ++j) {
			float sum = 0.0f;
			for (int k = 0; k < wA; ++k) {
				sum += A[i * wA + k] * B[k * wB + j];
			}
			C[i * wB + j] = sum;
		}
	}
}

/**
 * Main Testing Function for Matrix Multiplication
 * Executes both CUDA and CPU implementations and compares results and performance
 */
int MatrixMultiply(int argc, char** argv, int block_size, const dim3& dimsA, const dim3& dimsB) {
	/* Allocate host memory for input matrices A and B using CUDA pinned memory
	 * This improves data transfer performance between host and device
	 */
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A;
	cudaMallocHost(&h_A, mem_size_A);
	unsigned int size_B = dimsB.x * dimsB.y;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B;
	cudaMallocHost(&h_B, mem_size_B);
	cudaStream_t stream;

	/* Initialize input matrices with test values
	 * Matrix A is filled with 1.0, Matrix B with 0.01
	 */
	const float valB = 0.01f;
	ConstantInit(h_A, size_A, 1.0f);
	ConstantInit(h_B, size_B, valB);

	/* Allocate device (GPU) memory for matrices */
	float* d_A, * d_B, * d_C;

	/* Allocate host memory for result matrices (GPU and CPU results)
	 * The output matrix C has dimensions (dimsB.x × dimsA.y)
	 */
	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	float* h_C;      /* For GPU results */
	float* h_CCPU;   /* For CPU results */
	cudaMallocHost(&h_C, mem_size_C);
	cudaMallocHost(&h_CCPU, mem_size_C);

	if (h_C == NULL || h_CCPU == NULL) {
		fprintf(stderr, "Failed to allocate host matrices!\n");
		exit(EXIT_FAILURE);
	}

	cudaMalloc(reinterpret_cast<void**>(&d_A), mem_size_A);
	cudaMalloc(reinterpret_cast<void**>(&d_B), mem_size_B);
	cudaMalloc(reinterpret_cast<void**>(&d_C), mem_size_C);
	
	/* Set up CUDA events for precise timing measurements */
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/* Create non-blocking CUDA stream for asynchronous operations */
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

	/* Copy input matrices from host to device memory asynchronously */
	cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream);

	/* Configure execution parameters for the CUDA kernel
	 * Set up the number of threads per block and blocks per grid
	 */
	dim3 threads(block_size, block_size);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

	/* Begin GPU computation */
	printf("Computing result using CUDA Kernel...\n");

	/* Perform warmup operation to initialize GPU and eliminate startup overhead */
	if (block_size == 16) {
		MatrixMulCUDA<BLOCK_SIZE_16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
	}
	else {
		MatrixMulCUDA<BLOCK_SIZE_32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
	}

	printf("done\n");
	cudaStreamSynchronize(stream);

	/* Begin performance measurement */
	cudaEventRecord(start, stream);

	/* Execute kernel multiple times for accurate performance measurement
	 * Run 300 iterations to average out any timing fluctuations
	 */
	int nIter = 300;

	for (int j = 0; j < nIter; j++) {
		if (block_size == 16) {
			MatrixMulCUDA<BLOCK_SIZE_16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
		}
		else {
			MatrixMulCUDA<BLOCK_SIZE_32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
		}
	}

	/* End performance measurement */
	cudaEventRecord(stop, stream);

	/* Wait for all operations to complete */
	cudaEventSynchronize(stop);

	/* Calculate elapsed time in milliseconds */
	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);

	/* Calculate and display performance metrics for GPU implementation
	 * Compute GFLOPs (billions of floating-point operations per second)
	 */
	float msecPerMatrixMul = msecTotal / nIter;
	double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
		static_cast<double>(dimsA.y) *
		static_cast<double>(dimsB.x);
	double gigaFlops =
		(flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf(
		"GPU Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
		" WorkgroupSize= %u threads/block\n",
		gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

	/* Copy result from device back to host memory */
	cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	/* Now perform CPU matrix multiplication for comparison */
	printf("Computing result using CPU...\n");
	
	clock_t cpu_start = clock();
	
	/* Run CPU version fewer times since it's much slower
	 * Use 3 iterations instead of 300 for reasonable runtime
	 */
	int cpu_nIter = 3;
	for (int j = 0; j < cpu_nIter; j++) {
		MatrixMulCPU(h_CCPU, h_A, h_B, dimsA.x, dimsA.y, dimsB.x);
	}
	
	clock_t cpu_end = clock();
	double cpu_msecTotal = 1000.0 * (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
	double cpu_msecPerMatrixMul = cpu_msecTotal / cpu_nIter;
	double cpu_gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (cpu_msecPerMatrixMul / 1000.0f);
	
	printf(
		"CPU Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
		cpu_gigaFlops, cpu_msecPerMatrixMul, flopsPerMatrixMul);
	
	/* Calculate and display the GPU speedup compared to CPU */
	double speedup = cpu_msecPerMatrixMul / msecPerMatrixMul;
	printf("GPU Speedup over CPU: %.2fx\n", speedup);

	printf("Checking computed result for correctness: ");
	bool correct = true;

	/* Verify result correctness using relative error formula:
	 * |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|> < epsilon
	 * where epsilon is a small value representing machine precision
	 */
	double eps = 1.e-6;  /* Threshold for acceptable error */

	for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
		double abs_err = fabs(h_C[i] - (dimsA.x * valB));
		double dot_length = dimsA.x;
		double abs_val = fabs(h_C[i]);
		double rel_err = abs_err / abs_val / dot_length;

		if (rel_err > eps) {
			printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
				i, h_C[i], dimsA.x * valB, eps);
			correct = false;
		}
	}

	/* Additional validation: compare CPU and GPU results directly
	 * This helps identify any inconsistencies between implementations
	 */
	for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
		double abs_err = fabs(h_C[i] - h_CCPU[i]);
		double abs_val = fabs(h_C[i]);
		double rel_err = abs_err / (abs_val > 1e-10 ? abs_val : 1e-10);

		if (rel_err > eps) {
			printf("CPU/GPU mismatch! GPU[%05d]=%.8f, CPU=%.8f, diff=%.8f\n",
				i, h_C[i], h_CCPU[i], abs_err);
			/* Don't fail the test for this, just report it */
		}
	}

	printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

	/* Free all allocated memory resources */
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	cudaFreeHost(h_CCPU);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaStreamDestroy(stream);

	return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char** argv)
{
	printf("[Matrix Multiply Using CUDA] - Starting...\n");

	int block_size = 32;

	dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
	dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
		dimsB.x, dimsB.y);

	int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);

	return matrix_result;
} 
																																			
																																																
																																																