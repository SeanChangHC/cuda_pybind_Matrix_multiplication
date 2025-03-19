cmake_minimum_required(VERSION 3.5)
project(final_project LANGUAGES CXX CUDA)

# Set module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Detect operating system
message(${CMAKE_SYSTEM_NAME})
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(WINDOWS TRUE)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(LINUX TRUE)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()

# Detect architecture
if (${CMAKE_CL_64})
  set(ARCH_64 TRUE)
  set(PROCESSOR_ARCH "x64")
  message(64)
endif(${CMAKE_CL_64})

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA 10.1 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lnvToolExt -gencode arch=compute_52,code=sm_52)

# System-specific settings
if (WINDOWS)
  set(CUDA_TOOLKIT_SAMPLES_DIR "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1")
  add_definitions(-DBUILD_THIRDPARTY:BOOL=ON)
else()
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  if(COMPILER_SUPPORTS_CXX11)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
  elseif(COMPILER_SUPPORTS_CXX0X)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  else()
      message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
  endif()
  
  set(CUDA_TOOLKIT_SAMPLES_DIR ${CUDA_TOOLKIT_ROOT_DIR}/samples)
  set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR})
endif()

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11 installed via pip
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# Find additional dependencies
find_package(NVTX REQUIRED)
find_package(GLEW REQUIRED)
find_package(FREEGLUT REQUIRED)

# Add the CUDA kernel file
add_library(matrix_mul_kernel OBJECT matrix_mul_kernel.cu)
set_target_properties(matrix_mul_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create the pybind11 module
pybind11_add_module(cuda_matrix_mul 
    matrix_mul_wrapper.cpp
    $<TARGET_OBJECTS:matrix_mul_kernel>
)

# Link with CUDA libraries
target_link_libraries(cuda_matrix_mul PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# Set the output directory for the Python module
set_target_properties(cuda_matrix_mul PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Add src directory if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Src")
    add_subdirectory(Src)
endif() 