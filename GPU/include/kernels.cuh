#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#define N_THREADS 1024


__global__ void updateClassHV(float* input_hv, float* weights, int guess, int correct_class, int dim, float learning_rate);
__global__ void normMatRow(float* result, float* inputMat, int setNum, int colNum);
__global__ void cosineSimilarityVec(float* result, float* norm_1, int colNum, int dataidx);
__global__ void guessVecGenCompareCosine(bool* scoreboard, int* label, float* weights_norm, 
            const float * guess_table, const int setNum, const int n_class);
__global__ void guessVecGenCompareDot(bool* scoreboard, int* label,
    const float * guess_table, const int setNum, const int n_class);
__global__ void hd_init(float* results, float* hv, int* label, int Row, int Col, int maxVal);

#endif
