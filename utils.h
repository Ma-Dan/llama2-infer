#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <memory.h>
#include <stdlib.h>

using namespace std;

#define SAFE_DELETE(x) if(x){delete x; x=NULL;}

vector<string> split(const string& str, const string& delim);

bool is_same_shape(const vector<int> shape1, const vector<int> shape2);

#endif