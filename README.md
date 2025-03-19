# CUDA Matrix Multiplication with Python Bindings

This project demonstrates how to use pybind11 to create Python bindings for a CUDA matrix multiplication implementation.

## Requirements

- CUDA Toolkit (10.0+)
- CMake (3.18+)
- Python 3.6+
- pybind11
- numpy

## Installation

1. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

2. Install pybind11 for CMake:

```bash
pip install pybind11
```

3. Build the CUDA module:

```bash
mkdir build
cd build
cmake ..
make
```

This will generate a shared library file (e.g., `cuda_matrix_mul.so` on Linux/macOS or `cuda_matrix_mul.pyd` on Windows) in the project root directory.

## Usage

After building, you can import the module in Python:

```python
import numpy as np
import cuda_matrix_mul

# Create some matrices
A = np.random.rand(1024, 1024).astype(np.float32)
B = np.random.rand(1024, 1024).astype(np.float32)

# Multiply them using CUDA
C = cuda_matrix_mul.matrix_multiply(A, B)
```

## Example

Run the provided example:

```bash
python example.py
```

This will compare the performance of the CUDA matrix multiplication with NumPy's implementation.

## Notes

- The matrices must have dimensions that are multiples of the block size (16 or 32)
- Only single precision (float32) is supported
- Make sure to adjust the CUDA compute capability in CMakeLists.txt to match your GPU 