# 设置cmake版本，如果cmake版本过高，可能会出现错误
cmake_minimum_required(VERSION 2.8.12)
project(llama2-infer)

find_package(CUDA QUIET REQUIRED)

# 设置C++编译版本
set(CMAKE_CXX_STANDARD 17)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -g -O3 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86
    )

# Specify include directories
include_directories(
    kernels
    utility
    )

# 设置程序编译出的可执行文件
set(MAIN_FILE inference.cpp)
set(EXECUTABLE_OUTPUT_PATH ../)

add_compile_options(-g -Ofast)

cuda_add_executable(inference inference.cpp utils.cpp tensor.cpp node.cpp layer.cpp layer_register.cpp graph.cpp input.cpp embed.cpp unaryop.cpp binaryop.cpp reduction.cpp memorydata.cpp matmul.cpp reshape.cpp posenc.cpp concat.cpp softmax.cpp swish.cpp cuda_function.cu)
target_link_directories(inference PUBLIC /usr/local/cuda/lib64 /opt/cuda/lib64 $(CUDA_PATH)/targets/x86_64-linux/lib)
target_link_libraries(inference cublas culibos cudart cublasLt)
