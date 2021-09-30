
#include <iostream>
#include <hls_stream.h>
#include <ap_int.h>

using namespace std;

#define N_FEAT			617	//feature per input (e.g., isolet: 617)
#define N_CLASS			26	//number of classes. (e.g., isolet: 26, ucihar 12)
#define Dhv				2048 //hypervectors length
#define COL				8 //number of columns of a matrix-vector multiplication window (keep fixed 8)
#define ROW				32 //number of rows of a matrix-vector multiplication window (32, 64, 128, 256, 512)


#define PAD_			(N_FEAT & (COL - 1))
#if PAD_ == 0
	#define PAD 		0
#else
	#define PAD 		(COL - PAD_)
#endif

#define N_FEAT_PAD		(N_FEAT + PAD)	//feature per input (e.g., isolet: 624, ucihar 568)

typedef ap_int<2> dt_int2;

