#ifndef UI_VECTOR_MYLOGGER_H
#define UI_VECTOR_MYLOGGER_H

#include <iostream>
#include <fstream>
#include "RC.h"

using namespace std;

class myLogger {
public:
    myLogger();


    explicit myLogger(ofstream* ofs);

public:
    RC log(RC code, Level level, const char* const& srcfile, const char* const& function, int line);


    RC log(RC code, Level level);


    static const char* GetErrorString(RC code);


    static const char* GetLevelString(Level level);


    ~myLogger();

private:
    ofstream* ofs;
};


#endif //UI_VECTOR_MYLOGGER_H
