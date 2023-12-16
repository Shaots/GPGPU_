cmake_minimum_required(VERSION 3.27)
project(GPGPU_ CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(GPGPU_ main.cu test.cuh test.cu myVector.cuh myMatrix.cuh
        RC.h myLogger.h myLogger.cpp)


set_target_properties(GPGPU_ PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
