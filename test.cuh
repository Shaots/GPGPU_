#ifndef CPGPU_TEST_CUH
#define CPGPU_TEST_CUH

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <random>
#include "myVector.cuh"
#include "myMatrix.cuh"
#include "myLogger.h"

class test {
public:
    static void cpu_testVectorAdd(myLogger* logger);


    static void gpu_testVectorAdd1(myLogger* logger);


    static void gpu_testVectorAdd2(myLogger* logger);


    static void cpu_testMatrixMultiply(myLogger* logger);


    static void gpu_noShared_testMatrixMultiply1(myLogger* logger);


    static void gpu_noShared_testMatrixMultiply2(myLogger* logger);


    static void gpu_shared_testMatrixMultiply(myLogger* logger);


    static void allTest(myLogger* logger);
};


#endif //CPGPU_TEST_CUH
