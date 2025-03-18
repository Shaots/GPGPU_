#include "test.cuh"


void test::cpu_testVectorAdd(myLogger* logger) {
    std::cout << "CPU: test vector addition" << std::endl;
    myVector<int>::setLogger(logger);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const size_t len = 40000000;
    auto* v1 = new myVector<int>(len);
    auto* v2 = new myVector<int>(len);
    auto* v3 = new myVector<int>(len);
    int val1 = -2;
    int val2 = 4;
    for (size_t i = 0; i < len; ++i) {
        v1->setCoord(i, val1);
        v2->setCoord(i, val2);
    }
    std::cout << "Vector v1: length(v1) = " << v1->getLen() << "    v1_{i} = " << val1 << std::endl;
    std::cout << "Vector v2: length(v2) = " << v2->getLen() << "    v2_{i} = " << val2 << std::endl;

    size_t times = 10;
    cudaEventRecord(start, nullptr);
    for (size_t i = 0; i < times; ++i) {
        myVector<int>::addVec(v1, v2, v3);
    }

    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "v3 = v1 + v2" << std::endl;
    std::cout << "Vector v3: length(v3) = " << v3->getLen() << "    v3_{i} = " << v3->getCoord(0) << std::endl;
    //myVector::paintVector(v3);

    printf("Time spent executing by the CPU: %.2f ms\n", elapsedTime/(float)times);
    delete v1;
    delete v2;
    delete v3;
    std::cout << std::endl;
}


void test::gpu_testVectorAdd1(myLogger* logger) {
    std::cout << "GPU: test1 vector addition" << std::endl;
    myVector<int>::setLogger(logger);

    const size_t len = 10;
    auto* v1 = new myVector<int>(len);
    auto* v2 = new myVector<int>(len);
    auto* v3 = new myVector<int>(len);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, 99);
    for (size_t i = 0; i < len; ++i) {
        v1->setCoord(i, dist(gen));
        v2->setCoord(i, dist(gen));
    }
    std::cout << "Vector v1 = ";
    myVector<int>::paintVector(v1);
    std::cout << "Vector v2 = ";
    myVector<int>::paintVector(v2);
    int* dev_data1 = nullptr;
    int* dev_data2 = nullptr;
    int* dev_data3 = nullptr;


    cudaMalloc((void**) &dev_data1, len * sizeof(int));
    cudaMalloc((void**) &dev_data2, len * sizeof(int));
    cudaMalloc((void**) &dev_data3, len * sizeof(int));
    cudaMemcpy(dev_data1, v1->getData(), len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data2, v2->getData(), len * sizeof(int), cudaMemcpyHostToDevice);


    size_t blockLen = 512;
    size_t gridLen = len / blockLen + 1 * (len % blockLen != 0);
    dim3 blocksPerGrid = dim3(gridLen);
    dim3 threadsPerBlock = dim3(blockLen);
    std::cout << "blocksPerGrid = " << blocksPerGrid.x << "    " << "threadsPerBlock = " << threadsPerBlock.x
              << std::endl;

    ker_addVec<<<blocksPerGrid, threadsPerBlock>>>(dev_data1, dev_data2, dev_data3, len);

    auto* temp = new int[len];
    cudaMemcpy(temp, dev_data3, len * sizeof(int), cudaMemcpyDeviceToHost);
    v3->setData(temp, len);

    std::cout << "v3 = v1 + v2" << std::endl;
    std::cout << "Vector v3 = ";
    myVector<int>::paintVector(v3);

    delete v1;
    delete v2;
    delete v3;
    delete[] temp;
    cudaFree(dev_data1);
    cudaFree(dev_data2);
    cudaFree(dev_data3);
    std::cout << std::endl;
}


void test::gpu_testVectorAdd2(myLogger* logger) {
    std::cout << "GPU: test2 vector addition" << std::endl;
    myVector<int>::setLogger(logger);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    const size_t len = 40000000;
    auto* v1 = new myVector<int>(len);
    auto* v2 = new myVector<int>(len);
    auto* v3 = new myVector<int>(len);
    int val1 = -2.0;
    int val2 = 4.0;
    for (size_t i = 0; i < len; ++i) {
        v1->setCoord(i, val1);
        v2->setCoord(i, val2);
    }
    std::cout << "Vector v1: length(v1) = " << v1->getLen() << "    v1_{i} = " << val1 << std::endl;
    std::cout << "Vector v2: length(v2) = " << v2->getLen() << "    v2_{i} = " << val2 << std::endl;
    int* dev_data1 = nullptr;
    int* dev_data2 = nullptr;
    int* dev_data3 = nullptr;


    cudaMalloc((void**) &dev_data1, len * sizeof(int));
    cudaMalloc((void**) &dev_data2, len * sizeof(int));
    cudaMalloc((void**) &dev_data3, len * sizeof(int));
    cudaMemcpy(dev_data1, v1->getData(), len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data2, v2->getData(), len * sizeof(int), cudaMemcpyHostToDevice);

    size_t blockLen = 1024;
    size_t gridLen = len / blockLen + 1 * (len % blockLen != 0);
    dim3 blocksPerGrid = dim3(gridLen);
    dim3 threadsPerBlock = dim3(blockLen);
    std::cout << "blocksPerGrid = " << blocksPerGrid.x << "    " << "threadsPerBlock = " << threadsPerBlock.x
              << std::endl;

    cudaEventRecord(start, nullptr);
    ker_addVec<<<blocksPerGrid, threadsPerBlock>>>(dev_data1, dev_data2, dev_data3, len);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto* temp = new int[len];
    cudaMemcpy(temp, dev_data3, len * sizeof(int), cudaMemcpyDeviceToHost);
    v3->setData(temp, len);

    std::cout << "v3 = v1 + v2" << std::endl;
    std::cout << "Vector v3: length(v3) = " << v3->getLen() << "    v3_{i} = " << v3->getCoord(0) << std::endl;
    //myVector::paintVector(v3);
    printf("Time spent executing by the GPU: %.2f ms\n", elapsedTime);

    delete v1;
    delete v2;
    delete v3;
    delete[] temp;
    cudaFree(dev_data1);
    cudaFree(dev_data2);
    cudaFree(dev_data3);
    std::cout << std::endl;
}


void test::cpu_testMatrixMultiply(myLogger* logger) {
    std::cout << "CPU: test matrix multiply" << std::endl;
    myMatrix<int>::setLogger(logger);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const size_t row = 1500;
    const size_t len = 2000;
    const size_t col = 2500;

    auto* m1 = new myMatrix<int>(row, len);
    auto* m2 = new myMatrix<int>(len, col);
    auto* m3 = new myMatrix<int>(row, col);

    const int val1 = -2.0;
    for (size_t r = 0; r < row; ++r) {
        for (size_t c = 0; c < len; ++c) {
            m1->setCoord(r, c, val1);
        }
    }
    std::cout << "Matrix M1: size(M1) = (" << m1->getRow() << ", " << m1->getCol() << ")" << "    M1_{ij} = " << val1
              << std::endl;

    const int val2 = 3.0;
    for (size_t r = 0; r < len; ++r) {
        for (size_t c = 0; c < col; ++c) {
            m2->setCoord(r, c, val2);
        }
    }
    std::cout << "Matrix M2: size(M2) = (" << m2->getRow() << ", " << m2->getCol() << ")" << "    M2_{ij} = " << val2
              << std::endl;

    size_t times = 5;
    cudaEventRecord(start, nullptr);
    for (size_t i = 0; i < times; ++i) {
        myMatrix<int>::multiMatrix(m1, m2, m3);
    }
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "M3 = M1 * M2" << std::endl;
    std::cout << "Matrix M3: size(M3) = (" << m3->getRow() << ", " << m3->getCol() << ")" << "    M3_{ij} = "
              << m3->getCoord(0, 0) << std::endl;
    printf("Time spent executing by the CPU: %.2f ms\n", elapsedTime / (float) times);
    delete m1;
    delete m2;
    delete m3;
    std::cout << std::endl;
}


void test::gpu_noShared_testMatrixMultiply1(myLogger* logger) {
    std::cout << "GPU: test1 matrix multiply without shared memory" << std::endl;
    myMatrix<int>::setLogger(logger);

    const size_t row = 4;
    const size_t len = 5;
    const size_t col = 6;
    const size_t sz1 = row * len;
    const size_t sz2 = len * col;
    const size_t sz3 = row * col;


    auto* m1 = new myMatrix<int>(row, len);
    auto* m2 = new myMatrix<int>(len, col);
    auto* m3 = new myMatrix<int>(row, col);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, 9);

    for (size_t r = 0; r < row; ++r) {
        for (size_t c = 0; c < len; ++c) {
            m1->setCoord(r, c, dist(gen));
        }
    }
    std::cout << "Matrix M1: " << std::endl;
    myMatrix<int>::paintMatrix(m1);

    for (size_t r = 0; r < len; ++r) {
        for (size_t c = 0; c < col; ++c) {
            m2->setCoord(r, c, dist(gen));
        }
    }
    std::cout << "Matrix M2: " << std::endl;
    myMatrix<int>::paintMatrix(m2);

    int* dev_data1 = nullptr;
    int* dev_data2 = nullptr;
    int* dev_data3 = nullptr;


    cudaMalloc((void**) &dev_data1, sz1 * sizeof(int));
    cudaMalloc((void**) &dev_data2, sz2 * sizeof(int));
    cudaMalloc((void**) &dev_data3, sz3 * sizeof(int));
    cudaMemcpy(dev_data1, m1->getData(), sz1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data2, m2->getData(), sz2 * sizeof(int), cudaMemcpyHostToDevice);

    // row <--> y
    // col <--> x
    size_t blockX = 32;
    size_t blockY = 32;
    size_t gridX = col / blockX + 1 * (col % blockX != 0);
    size_t gridY = row / blockY + 1 * (row % blockY != 0);
    dim3 threadsPerBlock = dim3(blockX, blockY);
    dim3 blocksPerGrid = dim3(gridX, gridY);
    std::cout << "blocksPerGrid = (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")" << std::endl;
    std::cout << "threadsPerBlock = (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << std::endl;

    ker_multiMatrix_noShared<<<blocksPerGrid, threadsPerBlock>>>(dev_data1, dev_data2, dev_data3, row, len, col);

    auto* temp = new int[sz3];
    cudaMemcpy(temp, dev_data3, sz3 * sizeof(int), cudaMemcpyDeviceToHost);
    m3->setData(row, col, temp);

    std::cout << "M3 = M1 * M2:" << std::endl;
    myMatrix<int>::paintMatrix(m3);
    delete m1;
    delete m2;
    delete m3;
    delete[] temp;
    cudaFree(dev_data1);
    cudaFree(dev_data2);
    cudaFree(dev_data3);
    std::cout << std::endl;
}


void test::gpu_noShared_testMatrixMultiply2(myLogger* logger) {
    std::cout << "GPU: test2 matrix multiply without shared memory" << std::endl;
    myMatrix<int>::setLogger(logger);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    const size_t row = 1500;
    const size_t len = 2000;
    const size_t col = 2500;

    const size_t sz1 = row * len;
    const size_t sz2 = len * col;
    const size_t sz3 = row * col;


    auto* m1 = new myMatrix<int>(row, len);
    auto* m2 = new myMatrix<int>(len, col);
    auto* m3 = new myMatrix<int>(row, col);

    const int val1 = (int) -2.0;
    for (size_t r = 0; r < row; ++r) {
        for (size_t c = 0; c < len; ++c) {
            m1->setCoord(r, c, val1);
        }
    }
    std::cout << "Matrix M1: size(M1) = (" << m1->getRow() << ", " << m1->getCol() << ")" << "    M1_{ij} = " << val1
              << std::endl;

    const int val2 = (int) 3.0;
    for (size_t r = 0; r < len; ++r) {
        for (size_t c = 0; c < col; ++c) {
            m2->setCoord(r, c, val2);
        }
    }
    std::cout << "Matrix M2: size(M2) = (" << m2->getRow() << ", " << m2->getCol() << ")" << "    M2_{ij} = " << val2
              << std::endl;

    int* dev_data1 = nullptr;
    int* dev_data2 = nullptr;
    int* dev_data3 = nullptr;


    cudaMalloc((void**) &dev_data1, sz1 * sizeof(int));
    cudaMalloc((void**) &dev_data2, sz2 * sizeof(int));
    cudaMalloc((void**) &dev_data3, sz3 * sizeof(int));
    cudaMemcpy(dev_data1, m1->getData(), sz1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data2, m2->getData(), sz2 * sizeof(int), cudaMemcpyHostToDevice);

    // row <--> y
    // col <--> x
    size_t blockX = 32;
    size_t blockY = 32;
    size_t gridX = col / blockX + 1 * (col % blockX != 0);
    size_t gridY = row / blockY + 1 * (row % blockY != 0);
    dim3 threadsPerBlock = dim3(blockX, blockY);
    dim3 blocksPerGrid = dim3(gridX, gridY);
    std::cout << "blocksPerGrid = (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")" << std::endl;
    std::cout << "threadsPerBlock = (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << std::endl;

    size_t times = 50;
    cudaEventRecord(start, nullptr);
    for (size_t i = 0; i < times; ++i) {
        ker_multiMatrix_noShared<<<blocksPerGrid, threadsPerBlock>>>(dev_data1, dev_data2,
                                                                     dev_data3, row, len, col);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto* temp = new int[sz3];
    cudaMemcpy(temp, dev_data3, sz3 * sizeof(int), cudaMemcpyDeviceToHost);
    m3->setData(row, col, temp);
    //myMatrix::paintMatrix(m3);

    std::cout << "M3 = M1 * M2" << std::endl;
    std::cout << "Matrix M3: size(M3) = (" << m3->getRow() << ", " << m3->getCol() << ")" << "    M3_{ij} = "
              << m3->getCoord(0, 0) << std::endl;
    printf("Time spent executing by the GPU: %.2f ms\n", elapsedTime / (float) times);
    delete m1;
    delete m2;
    delete m3;
    delete[] temp;
    cudaFree(dev_data1);
    cudaFree(dev_data2);
    cudaFree(dev_data3);
    std::cout << std::endl;
}


void test::gpu_shared_testMatrixMultiply(myLogger* logger) {
    std::cout << "GPU: test matrix multiply with shared memory" << std::endl;
    myMatrix<int>::setLogger(logger);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    const size_t row = 1500;
    const size_t len = 2000;
    const size_t col = 2500;
    const size_t sz1 = row * len;
    const size_t sz2 = len * col;
    const size_t sz3 = row * col;


    auto* m1 = new myMatrix<int>(row, len);
    auto* m2 = new myMatrix<int>(len, col);
    auto* m3 = new myMatrix<int>(row, col);

    const int val1 = -2;
    for (size_t r = 0; r < row; ++r) {
        for (size_t c = 0; c < len; ++c) {
            m1->setCoord(r, c, val1);
        }
    }
    std::cout << "Matrix M1: size(M1) = (" << m1->getRow() << ", " << m1->getCol() << ")" << "    M1_{ij} = " << val1
              << std::endl;

    const int val2 = 3;
    for (size_t r = 0; r < len; ++r) {
        for (size_t c = 0; c < col; ++c) {
            m2->setCoord(r, c, val2);
        }
    }
    std::cout << "Matrix M2: size(M2) = (" << m2->getRow() << ", " << m2->getCol() << ")" << "    M2_{ij} = " << val2
              << std::endl;

    int* dev_data1 = nullptr;
    int* dev_data2 = nullptr;
    int* dev_data3 = nullptr;


    cudaMalloc((void**) &dev_data1, sz1 * sizeof(int));
    cudaMalloc((void**) &dev_data2, sz2 * sizeof(int));
    cudaMalloc((void**) &dev_data3, sz3 * sizeof(int));
    cudaMemcpy(dev_data1, m1->getData(), sz1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data2, m2->getData(), sz2 * sizeof(int), cudaMemcpyHostToDevice);

    // row <--> y
    // col <--> x
    dim3 threadsPerBlock = dim3(BLOCK_ONE_DIM, BLOCK_ONE_DIM);
    dim3 blocksPerGrid = dim3(col / BLOCK_ONE_DIM + 1, row / BLOCK_ONE_DIM + 1);
    std::cout << "blocksPerGrid = (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")" << std::endl;
    std::cout << "threadsPerBlock = (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << std::endl;

    size_t times = 50;
    cudaEventRecord(start, nullptr);
    for (size_t i = 0; i < times; ++i) {
        ker_multiMatrix_shared<<<blocksPerGrid, threadsPerBlock>>>(dev_data1, dev_data2, dev_data3, row, len, col);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto* temp = new int[sz3];
    cudaMemcpy(temp, dev_data3, sz3 * sizeof(int), cudaMemcpyDeviceToHost);
    m3->setData(row, col, temp);
    //myMatrix::paintMatrix(m3);

    std::cout << "M3 = M1 * M2" << std::endl;
    std::cout << "Matrix M3: size(M3) = (" << m3->getRow() << ", " << m3->getCol() << ")" << "    M3_{ij} = "
              << m3->getCoord(500, 400) << std::endl;
    printf("Time spent executing by the GPU: %.2f ms\n", elapsedTime / (float) times);
    delete m1;
    delete m2;
    delete m3;
    delete[] temp;
    cudaFree(dev_data1);
    cudaFree(dev_data2);
    cudaFree(dev_data3);
    std::cout << std::endl;
}


void test::allTest(myLogger* logger) {

    test::gpu_testVectorAdd1(logger);

    test::cpu_testVectorAdd(logger);

    test::gpu_testVectorAdd2(logger);

    cudaDeviceSynchronize();

    test::gpu_noShared_testMatrixMultiply1(logger);

    test::cpu_testMatrixMultiply(logger);

    test::gpu_noShared_testMatrixMultiply2(logger);

    cudaDeviceSynchronize();

    test::gpu_shared_testMatrixMultiply(logger);
}





