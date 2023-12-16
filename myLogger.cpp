#include "myLogger.h"

myLogger::myLogger() : ofs() {}

myLogger::myLogger(ofstream* ofs) : ofs(ofs) {}

RC myLogger::log(RC code, Level level, const char* const& srcfile, const char* const& function, int line) {
    if (ofs->is_open()) {
        *ofs << srcfile << " " << function << " " << line << endl;
        *ofs << GetLevelString(level) << " ";
        *ofs << GetErrorString(code) << endl;
    } else {
        cout << srcfile << " " << function << " " << line << endl;
        cout << GetLevelString(level) << " ";
        cout << GetErrorString(code) << endl;
    }
    return code;
}

RC myLogger::log(RC code, Level level) {
    if (ofs->is_open()) {
        *ofs << GetLevelString(level) << " ";
        *ofs << GetErrorString(code) << endl;
    } else {
        cout << GetLevelString(level) << " ";
        cout << GetErrorString(code) << endl;
    }
    return code;
}

myLogger::~myLogger() {
    if (ofs->is_open())
        ofs->close();
    delete ofs;
}

const char* myLogger::GetErrorString(RC code) {
    switch (code) {
        case RC::UNKNOWN:
            return "Unknown error";
        case RC::SUCCESS:
            return "Success";
        case RC::INVALID_ARGUMENT:
            return "Argument is invalid";
        case RC::MISMATCHING_DIMENSIONS:
            return "Mismatching dimension";
        case RC::INDEX_OUT_OF_BOUND:
            return "Index is out of bound";
        case RC::INFINITY_OVERFLOW:
            return "Number is greater than infinity";
        case RC::NOT_NUMBER:
            return "Not a number";
        case RC::ALLOCATION_ERROR:
            return "Couldn't allocate new memory";
        case RC::NULLPTR_ERROR:
            return "Received nullptr";
        case RC::FILE_NOT_FOUND:
            return "Couldn't find file with corresponding name";
        case RC::IO_ERROR:
            return "Couldn't write/read to/from file";
        default:
            return "";
    }
}

const char* myLogger::GetLevelString(Level level) {
    switch (level) {
        case Level::SEVERE:
            return "SEVERE: ";
        case Level::WARNING:
            return "WARNING: ";
        case Level::INFO:
            return "INFO: ";
        default:
            return "";
    }
}
