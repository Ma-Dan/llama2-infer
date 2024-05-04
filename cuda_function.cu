#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>

cublasHandle_t handle;

void createCublas()
{
    cublasCreate(&handle);
}

void destroyCublas()
{
    cublasDestroy(handle);
}

int mallocGPUData(float** gpuData, int length)
{
    cudaMalloc(gpuData, length);
    return 0;
}

int uploadGPUData(void *scratchGpu, void *scratchCpu, int length)
{
    cudaMemcpyAsync(scratchGpu, scratchCpu, length, cudaMemcpyHostToDevice);
    return 0;
}

void freeGPUData(void *gpuData)
{
    cudaFree(gpuData);
}

// Cublas
void matmul_cublas(float* xout, float* x, float* w, float* bias, float *d_B, float *d_C, int n, int d)
{
    dim3 dimsA(n, d, 1);
    dim3 dimsB(1, n, 1);
    dim3 dimsC(dimsB.x, dimsA.y, 1);

    //int mem_size_A = n*d*sizeof(float);
    int mem_size_B = n*sizeof(float);
    int mem_size_C = d*sizeof(float);

    // copy host memory to device
    cudaMemcpyAsync(d_B, x, mem_size_B, cudaMemcpyHostToDevice);

    float beta = 0.0f;

    if(bias != NULL)
    {
        cudaMemcpyAsync(d_C, bias, mem_size_C, cudaMemcpyHostToDevice);
        beta = 1.0f;
    }

    // Calculate with Cublas
    const float alpha = 1.0f;

    cublasStatus_t status = cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, dimsA.y, dimsB.x,
        dimsA.x, &alpha, w, dimsA.x, d_B,
        dimsB.x, &beta, d_C, dimsC.y);

    // Copy result from device to host
    cudaMemcpyAsync(xout, d_C, mem_size_C, cudaMemcpyDeviceToHost);
}
