#ifndef RC_H_INCLUDED
#define RC_H_INCLUDED

enum class RC {
    UNKNOWN,
    SUCCESS,
    INVALID_ARGUMENT,
    MISMATCHING_DIMENSIONS,
    INDEX_OUT_OF_BOUND,
    INFINITY_OVERFLOW, // Number is greater than infinity
    NOT_NUMBER,
    ALLOCATION_ERROR, // Couldn't allocate new memory
    NULLPTR_ERROR, // Received nullptr
    FILE_NOT_FOUND, // Couldn't find file with corresponding name
    IO_ERROR, // Couldn't write/read to/from file
};


enum class Level {
    SEVERE,   // Critical error that prevents application from running further
    WARNING, // Non-critical error
    INFO     // Optional information
};

#endif // RC_H_INCLUDED
