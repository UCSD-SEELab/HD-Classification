#ifndef OPENHD_RAND_CUH_
#define OPENHD_RAND_CUH_

#include <curand_kernel.h>

// __device__ curandState_t* states[__D__];

__global__ void 
__launch_bounds__(1024)
init_rand(curandState* states, unsigned int D, int seed) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tidx < D) {
        curandState_t* s = new curandState_t;
        if (s != 0) {
            curand_init(seed, tidx, 0, &states[tidx]);
        }
        __syncthreads();
    }
}

__device__ float __draw_random_hypervector__(curandState* states, const int d) {
    curandState_t s = states[d];
    float val = curand_uniform(&s);
    states[d] = s;

    //0.2 * 2 = 0.4 => 0
    //0.6 * 2 = 1.2 => 1

    return (int(val * 2) - 1)? -1 : 1;
}

__global__ void draw_bases(float* __ARG__id_base, float* __ARG__level_base, curandState* states, unsigned int D) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= D)
        return;

    *__hvdim__(__ARG__id_base, idx) = __draw_random_hypervector__(states, idx);
    *__hvdim__(__ARG__level_base, idx) = __draw_random_hypervector__(states, idx);
}


__global__ void draw_bases_dot(float* bases, curandState* states, unsigned int D, unsigned int rowNum) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= D)
        return;
    const int rowIdx = blockIdx.y;
    if (rowIdx >= rowNum)
        return;

    *__hvdim__(bases + rowIdx * D, idx) = __draw_random_hypervector__(states, idx);
}

__global__ void generate_hvs(float* id_hvs, float* id_base, unsigned int F, unsigned int D) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= D)
        return;
    
    #pragma unroll 1
    for(int f=0; f<F; ++f) {
        *__hxdim__(id_hvs, f, idx, D) = *__permute__(id_base, f, idx, D);
    }
}

#endif
