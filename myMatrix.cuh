#ifndef CPGPU_MYMATRIX_CUH
#define CPGPU_MYMATRIX_CUH

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <device_launch_parameters.h>
#include "myVector.cuh"
#include "myLogger.h"

template<typename T>
class myMatrix {
public:
    static myLogger* logger;

    static RC setLogger(myLogger* log) {
        if (log == nullptr) {
            return RC::NULLPTR_ERROR;
        }
        myMatrix<T>::logger = log;
        return RC::SUCCESS;
    }

public:
    static RC multiMatrix(myMatrix<T>* m1, myMatrix<T>* m2, myMatrix<T>* m3) {
        if (m1 == nullptr || m2 == nullptr || m3 == nullptr) {
            myMatrix<T>::logger->log(RC::NULLPTR_ERROR, Level::WARNING, __FILE__, __FUNCTION__, __LINE__);
            return RC::NULLPTR_ERROR;
        }
        if (m1->col != m2->row || m1->row != m3->row || m2->col != m3->col) {
            myMatrix<T>::logger->log(RC::MISMATCHING_DIMENSIONS, Level::WARNING, __FILE__, __FUNCTION__, __LINE__);
            return RC::MISMATCHING_DIMENSIONS;
        }
        size_t row = m1->row;
        size_t len = m1->col;
        size_t col = m2->col;
        for (size_t i = 0; i < row * col; ++i) {
            size_t r = i / col;
            size_t c = i % col;
            for (size_t j = 0; j < len; ++j) {
                m3->data[i] += m1->data[r * len + j] * m2->data[c + j * len];
            }
        }
        return RC::SUCCESS;
    }


    static RC paintMatrix(myMatrix<T>* matrix) {
        if (matrix == nullptr) {
            myMatrix<T>::logger->log(RC::NULLPTR_ERROR, Level::WARNING, __FILE__, __FUNCTION__, __LINE__);
            return RC::NULLPTR_ERROR;
        }
        size_t row = matrix->row;
        size_t col = matrix->col;
        T* data = matrix->data;
        for (size_t r = 0; r < row; ++r) {
            std::cout << "[";
            for (size_t c = 0; c < col - 1; ++c) {
                std::cout << data[r * col + c] << ", ";
            }
            std::cout << data[(r + 1) * col - 1] << "]" << std::endl;
        }
        return RC::SUCCESS;
    }

public:
    myMatrix(size_t row, size_t col) : row(row), col(col) {
        size_t sz = row * col;
        data = new(std::nothrow) T[sz];
        if (data == nullptr) {
            myMatrix<T>::logger->log(RC::ALLOCATION_ERROR, Level::SEVERE, __FILE__, __FUNCTION__, __LINE__);
        } else {
            for (size_t i = 0; i < sz; ++i) {
                data[i] = 0;
            }
        }
    }


    [[nodiscard]] size_t getRow() const {
        return row;
    }


    [[nodiscard]] size_t getCol() const {
        return col;
    }


    T const* getData() const {
        return data;
    }


    RC setData(size_t r, size_t c, T* data_) {
        if (r != row || c != col) {
            myMatrix<T>::logger->log(RC::MISMATCHING_DIMENSIONS, Level::WARNING, __FILE__, __FUNCTION__, __LINE__);
            return RC::MISMATCHING_DIMENSIONS;
        }
        memcpy(data, data_, r * c * sizeof(T));
        return RC::SUCCESS;
    }


    T getCoord(size_t r, size_t c) {
        if (r > row - 1 || c > col - 1) {
            myMatrix<T>::logger->log(RC::INDEX_OUT_OF_BOUND, Level::WARNING, __FILE__, __FUNCTION__, __LINE__);
            return NAN;
        }
        return data[col * r + c];
    }


    RC setCoord(size_t r, size_t c, T val) {
        if (r > row - 1 || c > col - 1) {
            myMatrix<T>::logger->log(RC::INDEX_OUT_OF_BOUND, Level::WARNING, __FILE__, __FUNCTION__, __LINE__);
            return RC::INDEX_OUT_OF_BOUND;
        }
        data[col * r + c] = val;
        return RC::SUCCESS;
    }


    ~myMatrix() {
        delete[] data;
    }

private:
    size_t row;
    size_t col;
    T* data;
};

template<typename T>
myLogger* myMatrix<T>::logger = nullptr;

template<typename T>
__global__ void
ker_multiMatrix_noShared(const T* data1, const T* data2, T* data3, size_t row, size_t len, size_t col) {
    // row <--> y
    // col <--> x
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int idx;
    if (x > col - 1 || y > row - 1)
        return;
    else {
        T sum = 0;
        unsigned int r = y * len;
        unsigned int c = x;
        for (size_t i = 0; i < len; ++i) {
            sum += data1[r + i] * data2[c + i * col];
        }
        idx = col * y + x;
        data3[idx] = sum;
    }
}


#define BLOCK_ONE_DIM 16

template<typename T>
__global__ void
ker_multiMatrix_shared(const T* data1, const T* data2, T* data3, size_t row, size_t len, size_t col) {
    // row <--> y
    // col <--> x

    __shared__ T sh1[BLOCK_ONE_DIM][BLOCK_ONE_DIM];
    __shared__ T sh2[BLOCK_ONE_DIM][BLOCK_ONE_DIM];
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    T sum = 0;
    unsigned int numSubMatrix = len / BLOCK_ONE_DIM + 1 * (len % BLOCK_ONE_DIM != 0);
    for (unsigned int k = 0; k < numSubMatrix; k++) {
        if (y > row - 1 || (threadIdx.x + k * BLOCK_ONE_DIM) > len - 1)
            sh1[threadIdx.y][threadIdx.x] = 0;
        else
            sh1[threadIdx.y][threadIdx.x] = data1[y * len + threadIdx.x + k * BLOCK_ONE_DIM];

        if (x > col - 1 || (threadIdx.y + k * BLOCK_ONE_DIM) > len - 1)
            sh2[threadIdx.y][threadIdx.x] = 0;
        else
            sh2[threadIdx.y][threadIdx.x] = data2[(threadIdx.y + k * BLOCK_ONE_DIM) * col + x];

        __syncthreads();

        for (unsigned int j = 0; j < BLOCK_ONE_DIM; j++)
            sum += sh1[threadIdx.y][j] * sh2[j][threadIdx.x];
        __syncthreads();
    }
    if (y < row && x < col)
        data3[y * col + x] = sum;
}


#endif //CPGPU_MYMATRIX_CUH
