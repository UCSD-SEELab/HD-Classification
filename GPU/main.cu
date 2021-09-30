#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include "include/openhd.cuh"
#include "include/openhd_rand.cuh"
#include "include/preprocessor.hpp"
#include "include/cudadebug.cuh"
#include "include/encoding.cuh"
#include "include/kernels.cuh"
#include <cublas_v2.h>
// #include <cuda_runtime.h>

#define MIN(x, y) ((x < y) ? x : y)

// #define USE_DOT_SIMILARITY
#define USE_COS_SIMILARITY

// #define USE_DOT_ENCODING
#define USE_LVID_ENCODING

__device__ int d_guess;

__global__ void findargmax(float* arr, int offset) {
    float max_val = arr[0];
    int max_idx = 0;
    for (int jj = 1; jj < offset; jj++) {
        if (max_val < arr[jj]) {
            max_val = arr[jj];
            max_idx = jj;
        }
    }
    d_guess = max_idx;
}

int main(int argc, char* argv[]) {
    // ./main [TRAIN dataset path] [TEST dataset path] [DIM] [ITER] [Learning Rate] [Q]
    // Example:
    // ./main datasets/UCIHAR/UCIHAR_train.choir_dat datasets/UCIHAR/UCIHAR_test.choir_dat 10000 20 1 100
    int nFeatures_train, nClasses_train;  // nFeatures is same as x_train[0].size()
    int nFeatures_test, nClasses_test;

    std::vector<std::vector<float>> x_test;
    std::vector<std::vector<float>> x_train;
    std::vector<int> y_train;
    std::vector<int> y_test;

    readChoirDat(argv[1], nFeatures_train, nClasses_train, x_train, y_train);
    readChoirDat(argv[2], nFeatures_test, nClasses_test, x_test, y_test);

    // normalize
    l2norm(x_train);
    l2norm(x_test);

    std::vector<float> x_train_flat = flatten(x_train);
    std::vector<float> x_test_flat = flatten(x_test);

    // base_creation: linear
    int dim = atoi(argv[3]);
    int iter_num = atoi(argv[4]);

    float learning_rate = (float)atof(argv[5]);

    int Q = atoi(argv[6]);

    int train_set_num = x_train.size();
    int test_set_num = x_test.size();

    std::cout << x_train.size() <<std::endl;
    std::cout << x_test.size() <<std::endl;

    int train_encode_size = train_set_num * dim;
    int test_encode_size = test_set_num * dim;

    //////////////////////////////////////////////////////////////////////////
    // GPU LOAD
    // int nThreads = N_THREADS;
    // int nBlocks = int(ceil(float(features) / float(nThreads)));
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    cudaEvent_t dataload, start, stop1, stop2, stop3;
    cudaError_t err = cudaSuccess;

    cudaEventCreate(&dataload);
    cudaEventCreate(&start);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);
    cudaEventCreate(&stop3);

    int* d_y_train = NULL;
    int* d_y_test = NULL;
    float* d_x_train = NULL;
    float* d_x_test = NULL;
    float* d_hvs_train = NULL;
    float* d_hvs_test = NULL;
#ifdef USE_COS_SIMILARITY
    float* d_weights_norm = NULL;
#endif

    cudaEventRecord(dataload);

    HANDLE_ERROR(cudaMalloc((void **)&d_y_train, y_train.size() * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&d_y_test, y_test.size() * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&d_x_train, x_train_flat.size() * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_x_test, x_test_flat.size() * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_hvs_train, 4 * train_encode_size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_hvs_test, 4 * test_encode_size * sizeof(float)));

#ifdef USE_COS_SIMILARITY
    HANDLE_ERROR(cudaMalloc((void **)&d_weights_norm, nClasses_train * sizeof(float)));
#endif

    HANDLE_ERROR(cudaMemcpy(d_y_train, y_train.data(), y_train.size() * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_y_test, y_test.data(), y_test.size() * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_x_train, x_train_flat.data(), x_train_flat.size() * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_x_test, x_test_flat.data(), x_test_flat.size() * sizeof(float), cudaMemcpyHostToDevice));


    curandState* devState;
    HANDLE_ERROR(cudaMalloc((void**)&devState, dim * sizeof(curandState)));
    dim3 nBlock(N_THREADS, 1, 1);
    dim3 nGrid((dim + N_THREADS - 1) / N_THREADS, 1);
    init_rand<<<nGrid, nBlock>>>(devState, dim, 0);

#ifdef USE_DOT_ENCODING
    float* d_bases = NULL;
    dim3 dotBaseblocks((dim + N_THREADS - 1) / N_THREADS, train_set_num);
    HANDLE_ERROR(cudaMalloc((void **)&d_bases, nFeatures_train * dim * sizeof(float)));
    draw_bases_dot<<<dotBaseblocks, nBlock>>>(d_bases, devState, dim, nFeatures_train);
#endif

#if defined(USE_LVID_ENCODING)
    float* d_level_hvs = NULL;
    float* d_id_hvs = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_level_hvs, (Q + 1) * dim * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_id_hvs, nFeatures_train * dim * sizeof(float)));

    float* d_id_base;
    float* d_level_base;
    HANDLE_ERROR(cudaMalloc((void **)&d_id_base, dim * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_level_base, dim * sizeof(float)));


    draw_bases<<<nGrid, nBlock>>>(d_id_base, d_level_base, devState, dim);
    generate_hvs<<<nGrid, nBlock>>>(d_id_hvs, d_id_base, nFeatures_train, dim);
    generate_hvs<<<nGrid, nBlock>>>(d_level_hvs, d_level_base, Q+1, dim);
#endif

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1;
    const float beta = 0;

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    printf("Starting Encoding Stage...\n");

#ifdef USE_DOT_ENCODING
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
               dim, train_set_num, nFeatures_train, 
               &alpha, d_bases, dim, 
               d_x_train, nFeatures_train, &beta, 
               d_hvs_train, dim);
    cudaDeviceSynchronize();
#endif
#ifdef USE_LVID_ENCODING
    dim3 encodeblocksTrain((dim + N_THREADS - 1) / N_THREADS, train_set_num);
    dim3 encodeTPB(N_THREADS, 1, 1);

    int level_stride = dim * 4;
    int id_stride = dim * 4;
    int fm_stride = nFeatures_train * 4;
    encodeLevelId<<<encodeblocksTrain, encodeTPB>>>(d_level_hvs, d_id_hvs, d_x_train, d_hvs_train, level_stride, 
                                                id_stride, fm_stride, train_set_num, nFeatures_train, Q, dim);
#endif

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);

    // Training stage
    // TODO: Add validation
    // make_guess and create guess table
    printf("Training stage...\n");

    float* d_guess_vec = NULL;
    float* d_weights = NULL;

    int guess_hit_training = 0;

//     HANDLE_ERROR(cudaMalloc((void **)&d_guess_vec, nClasses_train * sizeof(float)));  // show prob. for class for each case
//     HANDLE_ERROR(cudaMalloc((void **)&d_weights, nClasses_train * dim * sizeof(float)));
//     HANDLE_ERROR(cudaMemset(d_weights, 0, nClasses_train * dim * sizeof(float)));

// #ifdef USE_COS_SIMILARITY
//     HANDLE_ERROR(cudaMemset(d_weights_norm, 0, nClasses_train * sizeof(float)));
// #endif

    HANDLE_ERROR(cudaMalloc((void **)&d_guess_vec, nClasses_train * sizeof(float)));  // show prob. for class for each case
    HANDLE_ERROR(cudaMalloc((void **)&d_weights, nClasses_train * dim * sizeof(float)));
    // initialize hd model single pass training
    const int small_threads = 32;  // FIXME
    int blocksX = MIN((train_set_num + small_threads - 1)/small_threads, p.maxGridSize[0]);
    int blocksY = MIN(dim, p.maxGridSize[1]);
    dim3 nblocks(blocksX, blocksY, 1);  //C, D
    dim3 nthreads(small_threads, 1, 1);

    hd_init<<<nblocks, nthreads, small_threads*sizeof(float)>>>(d_weights, d_hvs_train, d_y_train, train_set_num, dim, nClasses_train);

#ifdef USE_COS_SIMILARITY
    // HANDLE_ERROR(cudaMemset(d_weights_norm, 0, nClasses_train * sizeof(float)));
    normMatRow<<<(nClasses_train+N_THREADS-1)/N_THREADS, N_THREADS>>>(d_weights_norm, d_weights, nClasses_train, dim);
#endif



#pragma unroll
    for (int iter = 0; iter < iter_num; ++iter) {  // Retraining
#pragma unroll
        for (int ii = 0; ii < train_set_num; ++ii) {
            // TODO: Implement batch
            cublasSgemv(handle, CUBLAS_OP_T, 
                        dim, nClasses_train, 
                        &alpha, d_weights, dim, 
                        d_hvs_train + ii * dim, 1, &beta, 
                        d_guess_vec, 1);   // np.dot
            cudaDeviceSynchronize();

#ifdef USE_COS_SIMILARITY
            cosineSimilarityVec<<<(nClasses_train + N_THREADS - 1)/N_THREADS, N_THREADS>>>(d_guess_vec, d_weights_norm, nClasses_train, ii);
#endif

            int guess = -1;
            findargmax<<<1, 1>>>(d_guess_vec, nClasses_train);
            cudaMemcpyFromSymbol(&guess, d_guess, sizeof(int), 0, cudaMemcpyDeviceToHost);
            
            // update weight matrix
            if (guess != y_train[ii]) {
                updateClassHV<<<(dim + N_THREADS - 1)/N_THREADS, N_THREADS>>>(d_hvs_train + ii * dim, d_weights, guess, y_train[ii], dim, learning_rate);
#ifdef USE_COS_SIMILARITY
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
                cublasSnrm2(handle, dim, d_weights + y_train[ii] * dim, 1, d_weights_norm + y_train[ii]);
                // cudaMemcpy(d_weights_norm + guess, d_weights_norm + y_train[ii], sizeof(float), cudaMemcpyDeviceToDevice);
                cublasSnrm2(handle, dim, d_weights + guess * dim, 1, d_weights_norm + guess);
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
#endif
            }
            else
                guess_hit_training++;
        }
    }
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    // Testing stage: d_hvs_test vs d_weights (classes * dim)
    printf("Starting Test Stage..\n");

#ifdef USE_DOT_ENCODING
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
               dim, test_set_num, nFeatures_test, 
               &alpha, d_bases, dim, 
               d_x_test, nFeatures_test, &beta, 
               d_hvs_test, dim);
    cudaDeviceSynchronize();
#endif
#ifdef USE_LVID_ENCODING
    dim3 encodeblocksTest((dim + N_THREADS - 1) / N_THREADS, test_set_num);
    encodeLevelId<<<encodeblocksTest, encodeTPB>>>(d_level_hvs, d_id_hvs, d_x_test, d_hvs_test, level_stride, 
                                                id_stride, fm_stride, test_set_num, nFeatures_test, Q, dim);
#endif

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float* d_guess_vec_test = NULL;
    bool* d_scoreboard = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_guess_vec_test, test_set_num * nClasses_test * sizeof(float)));  // show prob. for class for each case
    HANDLE_ERROR(cudaMalloc((void **)&d_scoreboard, test_set_num * sizeof(bool)));

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        nClasses_test, test_set_num, dim, 
        &alpha, d_weights, dim, 
        d_hvs_test, dim, &beta, 
        d_guess_vec_test, nClasses_test);
    cudaDeviceSynchronize();

#ifdef USE_COS_SIMILARITY
    guessVecGenCompareCosine<<<(test_set_num + N_THREADS - 1) / N_THREADS, N_THREADS>>>
        (d_scoreboard, d_y_test, d_weights_norm, d_guess_vec_test, test_set_num, nClasses_test);
#endif
#ifdef USE_DOT_SIMILARITY
    guessVecGenCompareDot<<<(test_set_num + N_THREADS - 1) / N_THREADS, N_THREADS>>>
        (d_scoreboard, d_y_test, d_guess_vec_test, test_set_num, nClasses_test);
#endif
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);

    std::cout << "Train Acc: " << (float) guess_hit_training/(train_set_num * iter_num) << std::endl;
    
    int guess_hit_testing = 0;
    bool* scoreboard = (bool*)malloc(test_set_num * sizeof(bool));
    HANDLE_ERROR(cudaMemcpy(scoreboard, d_scoreboard, test_set_num * sizeof(bool), cudaMemcpyDeviceToHost));
    for (int jj = 0; jj < test_set_num; ++jj) {
        if (scoreboard[jj] == 1)
            guess_hit_testing++;
    }
    std::cout << "Test Acc: " << (float) guess_hit_testing/test_set_num << std::endl;

    cublasDestroy(handle);
#ifdef USE_DOT_ENCODING
    HANDLE_ERROR(cudaFree(d_bases));
#endif
    HANDLE_ERROR(cudaFree(d_y_train));
    HANDLE_ERROR(cudaFree(d_y_test));
    HANDLE_ERROR(cudaFree(d_x_train));
    HANDLE_ERROR(cudaFree(d_x_test));
    HANDLE_ERROR(cudaFree(d_hvs_train));
    HANDLE_ERROR(cudaFree(d_hvs_test));
    HANDLE_ERROR(cudaFree(d_guess_vec));
    HANDLE_ERROR(cudaFree(d_weights));

#ifdef USE_COS_SIMILARITY
    HANDLE_ERROR(cudaFree(d_weights_norm));
#endif

    HANDLE_ERROR(cudaFree(d_guess_vec_test));
    HANDLE_ERROR(cudaFree(d_scoreboard));

#if defined(USE_LVID_ENCODING)
    HANDLE_ERROR(cudaFree(d_level_hvs));
    HANDLE_ERROR(cudaFree(d_id_hvs));
    HANDLE_ERROR(cudaFree(d_level_base));
    HANDLE_ERROR(cudaFree(d_id_base));
#endif
    HANDLE_ERROR(cudaFree(devState));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, dataload, start);
    printf("GPU Execution time (Data Loading): %f\n", milliseconds);
    cudaEventElapsedTime(&milliseconds, start, stop1);
    printf("GPU Execution time (Encoding): %f\n", milliseconds);
    cudaEventElapsedTime(&milliseconds, stop1, stop2);
    printf("GPU Execution time (Training): %f\n", milliseconds);
    cudaEventElapsedTime(&milliseconds, stop2, stop3);
    printf("GPU Execution time (Inference): %f\n", milliseconds);
    cudaEventElapsedTime(&milliseconds, dataload, stop3);
    printf("GPU Execution time (End-to-end): %f\n", milliseconds);
    cudaEventElapsedTime(&milliseconds, start, stop3);
    printf("GPU Execution time (w/o load): %f\n", milliseconds);
    
    cudaEventDestroy(dataload);
    cudaEventDestroy(start);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);
    cudaEventDestroy(stop3);
    
    free(scoreboard);
    return 0;
}
