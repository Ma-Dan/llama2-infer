# 设置cmake版本，如果cmake版本过高，可能会出现错误
cmake_minimum_required(VERSION 2.8.12)
project(llama2-infer)

# 设置C++编译版本
set(CMAKE_CXX_STANDARD 17)

set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/re2)

# 设置程序编译出的可执行文件
set(MAIN_FILE inference.cpp)
set(EXECUTABLE_OUTPUT_PATH ../)

add_compile_options(-Ofast -fopenmp)

add_executable(inference inference.cpp utils.cpp tensor.cpp node.cpp layer.cpp layer_register.cpp graph.cpp input.cpp embed.cpp unaryop.cpp binaryop.cpp reduction.cpp memorydata.cpp matmul.cpp reshape.cpp posenc.cpp concat.cpp softmax.cpp swish.cpp)
target_link_libraries(inference omp5 re2::re2)
