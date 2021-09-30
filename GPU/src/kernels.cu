#include "../include/kernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

struct square { __host__ __device__ float operator()(float x) { return x * x; } };


__global__ void updateClassHV(float* input_hv, float* weights, int guess, int correct_class, int dim, float learning_rate) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < dim) {
        float input_hv_p = input_hv[idx];
        weights[guess * dim + idx] -= learning_rate * input_hv_p;
        weights[correct_class * dim + idx] += learning_rate * input_hv_p;
    }
}

__global__ void normMatRow(float* result, float* inputMat, int setNum, int colNum) {
    for (int rowNum = blockIdx.x * blockDim.x + threadIdx.x; 
        rowNum < setNum; 
        rowNum += blockDim.x * gridDim.x)
    {
        // result[rowNum] = normf(colNum, inputMat + rowNum * colNum);
        result[rowNum] = sqrt(thrust::transform_reduce(thrust::device, inputMat + rowNum * colNum, inputMat + (rowNum + 1) * colNum, square(), 0.0f, thrust::plus<float>()));
    }
}


__global__ void cosineSimilarityVec(float* result, float* norm_1, int colNum, int dataidx) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < colNum)
        result[idx] = result[idx] / (norm_1[idx]);
}

__global__
void guessVecGenCompareCosine(bool* scoreboard, int* label, float* weights_norm,
    const float * guess_table, const int setNum, const int n_class) {

    int rowNum = threadIdx.x + blockDim.x * blockIdx.x;

    if (rowNum < setNum) {
        float max_value = guess_table[n_class * rowNum + 0] / (weights_norm[0]);
        // float max_value = guess_table[n_class * rowNum + 0];
        int max_idx = 0;
        for (int j = 1; j < n_class; j++){
            float val_to_compare = guess_table[n_class * rowNum + j] / (weights_norm[j]);
            if (max_value < val_to_compare) {
                max_value = val_to_compare;
                max_idx = j;
            }
        }
        // scoreboard is initialized to 1 at first
        // we don't have to overwrite
        if (max_idx == label[rowNum])  //compare
            scoreboard[rowNum] = 1;
        else
            scoreboard[rowNum] = 0;
    }
}

__global__
void guessVecGenCompareDot(bool* scoreboard, int* label,
    const float * guess_table, const int setNum, const int n_class) {

    int rowNum = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowNum < setNum) {
        float max_value = guess_table[n_class * rowNum + 0];
        // float max_value = guess_table[n_class * rowNum + 0];
        int max_idx = 0;
        for (int j = 1; j < n_class; j++){
            float val_to_compare = guess_table[n_class * rowNum + j];
            if (max_value < val_to_compare) {
                max_value = val_to_compare;
                max_idx = j;
            }
        }
        // scoreboard is initialized to 1 at first
        // we don't have to overwrite
        if (max_idx == label[rowNum])  //compare
            scoreboard[rowNum] = 1;
        else
            scoreboard[rowNum] = 0;
    }
}

__global__ void hd_init(float* results, float* hv, int* label, int Row, int Col, int maxVal) {
    extern __shared__ float temp[];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = blockIdx.y;

    int t = threadIdx.x;

    temp[t] = 0;
    __syncthreads();


    while (j < Col) {
        while (i < Row) {
            atomicAdd(&temp[label[i]], hv[i*Col+j]);
            i += blockDim.x * gridDim.x;
        }
        __syncthreads();
        atomicAdd(&results[t*Col+j], temp[t]);

        j += blockDim.y * gridDim.y;
    }
}

