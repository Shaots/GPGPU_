#include "test.cuh"
#include "info.cuh"

int main() {
    const char* filename = "error.txt";
    auto* ofs = new ofstream (filename, ios::out);
    auto* logger = new myLogger(ofs);

    printInfoDevice();
    test::allTest(logger);

    delete logger;
    return 0;
}