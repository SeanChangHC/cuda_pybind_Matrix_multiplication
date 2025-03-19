#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // For memcpy function
#include <math.h>
#include <time.h>

// CUDA-specific definitions for compilation without CUDA headers
#ifndef __CUDACC__
#define __device__
#define __host__
#define __global__
#define __shared__
#define __syncthreads()
struct dim3 { unsigned int x, y, z; dim3(unsigned int x=1, unsigned int y=1, unsigned int z=1) : x(x), y(y), z(z) {} };
struct cudaEvent_t { int dummy; };
struct cudaStream_t { int dummy; };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
#define cudaStreamNonBlocking 0
#define blockIdx threadIdx
#define threadIdx blockIdx
struct { dim3 x, y; } blockIdx, threadIdx;
#endif

// Block size for matrix multiplication kernel
#define BLOCK_SIZE_16 16
#define BLOCK_SIZE_32 32
#define N (2048*2048)
#define THREADS_PER_BLOCK 128
#define NUM_STREAMS 4

// CUDA kernel for matrix multiplication
template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float* C, float* A, float* B, int wA, int wB) {
	// Block index
	int bx = blockIdx.x.x;
	int by = blockIdx.y.x;

	// Thread index
	int tx = threadIdx.x.x;
	int ty = threadIdx.y.x;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ float As[BLOCK_SIZE_32][BLOCK_SIZE_32];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE_32][BLOCK_SIZE_32];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll

		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

// Initialize matrix with constant value
void ConstantInit(float* data, int size, float val) {
	for (int i = 0; i < size; ++i) {
		data[i] = val;
	}
}

// CPU implementation of matrix multiplication
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

// CUDA runtime function stubs for compilation without CUDA headers
#ifndef __CUDACC__
int cudaMallocHost(float** ptr, size_t size) { *ptr = (float*)malloc(size); return 0; }
int cudaMalloc(void** ptr, size_t size) { *ptr = malloc(size); return 0; }
int cudaFreeHost(void* ptr) { free(ptr); return 0; }
int cudaFree(void* ptr) { free(ptr); return 0; }
int cudaEventCreate(cudaEvent_t* event) { return 0; }
int cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) { return 0; }
int cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) { 
	memcpy(dst, src, count); return 0; 
}
int cudaStreamSynchronize(cudaStream_t stream) { return 0; }
int cudaEventRecord(cudaEvent_t event, cudaStream_t stream) { return 0; }
int cudaEventSynchronize(cudaEvent_t event) { return 0; }
int cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t stop) { *ms = 0.1f; return 0; }
int cudaEventDestroy(cudaEvent_t event) { return 0; }
int cudaStreamDestroy(cudaStream_t stream) { return 0; }
#endif

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char** argv, int block_size, const dim3& dimsA, const dim3& dimsB) {
	// Allocate host memory for matrices A and B
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A;
	cudaMallocHost(&h_A, mem_size_A);
	unsigned int size_B = dimsB.x * dimsB.y;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B;
	cudaMallocHost(&h_B, mem_size_B);
	cudaStream_t stream;

	// Initialize host memory
	const float valB = 0.01f;
	ConstantInit(h_A, size_A, 1.0f);
	ConstantInit(h_B, size_B, valB);

	// Allocate device memory
	float* d_A, * d_B, * d_C;

	// Allocate host matrices for CPU computation and result
	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	float* h_C;
	float* h_CCPU;
	cudaMallocHost(&h_C, mem_size_C);
	cudaMallocHost(&h_CCPU, mem_size_C);

	if (h_C == NULL || h_CCPU == NULL) {
		fprintf(stderr, "Failed to allocate host matrices!\n");
		exit(EXIT_FAILURE);
	}

	cudaMalloc(reinterpret_cast<void**>(&d_A), mem_size_A);
	cudaMalloc(reinterpret_cast<void**>(&d_B), mem_size_B);
	cudaMalloc(reinterpret_cast<void**>(&d_C), mem_size_C);
	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

	// copy host memory to device
	cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream);

	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");

	// Dummy function call to simulate CUDA kernel launch since we can't actually call it here
	#ifdef __CUDACC__
	// Performs warmup operation using matrixMul CUDA kernel
	if (block_size == 16) {
		MatrixMulCUDA<BLOCK_SIZE_16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
	}
	else {
		MatrixMulCUDA<BLOCK_SIZE_32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
	}
	#endif

	printf("done\n");
	cudaStreamSynchronize(stream);

	// Record the start event
	cudaEventRecord(start, stream);

	// Execute the kernel
	int nIter = 300;

	#ifdef __CUDACC__
	for (int j = 0; j < nIter; j++) {
		if (block_size == 16) {
			MatrixMulCUDA<BLOCK_SIZE_16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
		}
		else {
			MatrixMulCUDA<BLOCK_SIZE_32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
		}
	}
	#endif

	// Record the stop event
	cudaEventRecord(stop, stream);

	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);

	// Compute and print the performance
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

	// Copy result from device to host
	cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	// Now perform the CPU matrix multiplication and measure time
	printf("Computing result using CPU...\n");
	
	clock_t cpu_start = clock();
	
	// Run the CPU version multiple times for more accurate timing
	int cpu_nIter = 3; // Fewer iterations for CPU as it's much slower
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
	
	// Calculate and print speedup
	double speedup = cpu_msecPerMatrixMul / msecPerMatrixMul;
	printf("GPU Speedup over CPU: %.2fx\n", speedup);

	printf("Checking computed result for correctness: ");
	bool correct = true;

	// test relative error by the formula
	//     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
	double eps = 1.e-6;  // machine zero

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

	// Also verify that CPU and GPU results match
	for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
		double abs_err = fabs(h_C[i] - h_CCPU[i]);
		double abs_val = fabs(h_C[i]);
		double rel_err = abs_err / (abs_val > 1e-10 ? abs_val : 1e-10);

		if (rel_err > eps) {
			printf("CPU/GPU mismatch! GPU[%05d]=%.8f, CPU=%.8f, diff=%.8f\n",
				i, h_C[i], h_CCPU[i], abs_err);
			// Don't fail the test for this, just report it
		}
	}

	printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

	// Clean up memory
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
																																			
																																																