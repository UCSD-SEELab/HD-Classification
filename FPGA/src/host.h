#include "host.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <vector>
#include <time.h>
#include <chrono>

using namespace std;

int train = 3;

#define N_CLASS		26	//number of classes. (e.g., isolet: 26, ucihar 12)
#define Dhv				2048  //hypervectors length
string X_train_path = "./isolet_trainX.bin";
string y_train_path = "./isolet_trainY.bin";
string X_test_path = "./isolet_testX.bin";
string y_test_path = "./isolet_testY.bin";


