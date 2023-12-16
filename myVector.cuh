#ifndef CPGPU_MYVECTOR_CUH
#define CPGPU_MYVECTOR_CUH


#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include <device_launch_parameters.h>
#include "myLogger.h"

template<typename T>
class myVector {
public:
    static myLogger* logger;


    static RC setLogger(myLogger* log) {
        if (log == nullptr) {
            return RC::NULLPTR_ERROR;
        }
        myVector<T>::logger = log;
        return RC::SUCCESS;
    }


    static RC paintVector(myVector<T>* vec) {
        if (vec == nullptr) {
            myVector<T>::logger->log(RC::NULLPTR_ERROR, Level::WARNING, __FILE__, __FUNCTION__, __LINE__);
            return RC::NULLPTR_ERROR;
        }
        size_t len = vec->len;
        T* data = vec->data;
        std::cout << "[";
        for (size_t i = 0; i < len - 1; ++i) {
            std::cout << data[i] << ", ";
        }
        std::cout << data[len - 1] << "]" << std::endl;
        return RC::SUCCESS;
    }

public:
    static RC addVec(myVector<T>* vec1, myVector<T>* vec2, myVector<T>* vec3) {
        if (vec1 == nullptr || vec2 == nullptr || vec3 == nullptr) {
            myVector<T>::logger->log(RC::NULLPTR_ERROR, Level::WARNING, __FILE__, __FUNCTION__, __LINE__);
            return RC::NULLPTR_ERROR;
        }
        if (vec1->len != vec2->len || vec2->len != vec3->len) {
            myVector<T>::logger->log(RC::MISMATCHING_DIMENSIONS, Level::WARNING, __FILE__, __FUNCTION__, __LINE__);
            return RC::MISMATCHING_DIMENSIONS;
        }
        size_t len = vec1->len;
        for (size_t i = 0; i < len; ++i) {
            vec3->data[i] = vec1->data[i] + vec2->data[i];
        }
        return RC::SUCCESS;
    }

public:
    explicit myVector(size_t len) : len(len), data(new(nothrow) int[len]) {
        if (this->data == nullptr) {
            myVector<T>::logger->log(RC::ALLOCATION_ERROR, Level::SEVERE, __FILE__, __FUNCTION__, __LINE__);
        } else {
            for (size_t i = 0; i < len; ++i) {
                data[i] = 0;
            }
        }
    }


    [[nodiscard]] size_t getLen() const {
        return len;
    }

    T const* getData() {
        return data;
    }


    RC setData(T* data_, size_t length) {
        if (length != len) {
            myVector::logger->log(RC::MISMATCHING_DIMENSIONS, Level::WARNING, __FILE__, __FUNCTION__, __LINE__);
            return RC::MISMATCHING_DIMENSIONS;
        }
        memcpy(data, data_, len * sizeof(T));
        return RC::SUCCESS;
    }


    T getCoord(size_t position) {
        if (position > len - 1) {
            myVector::logger->log(RC::INDEX_OUT_OF_BOUND, Level::WARNING, __FILE__, __FUNCTION__, __LINE__);
            return NAN;
        }
        return data[position];
    }


    RC setCoord(size_t position, T val) {
        if (position > len - 1) {
            myVector::logger->log(RC::INDEX_OUT_OF_BOUND, Level::WARNING, __FILE__, __FUNCTION__, __LINE__);
            return RC::INDEX_OUT_OF_BOUND;
        }
        data[position] = val;
        return RC::SUCCESS;
    }


    ~myVector() {
        delete[] data;
    }

private:
    size_t len;
    T* data;
};

template<typename T>
myLogger* myVector<T>::logger = nullptr;


template<typename T>
__global__ void ker_addVec(const T* data1, const T* data2, T* data3, size_t len) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > len - 1)
        return;
    data3[idx] = data1[idx] + data2[idx];
}

#endif //CPGPU_MYVECTOR_CUH
