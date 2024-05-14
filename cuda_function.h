void createCublas();
void destroyCublas();
int mallocGPUData(float** gpuData, int length);
int uploadGPUData(void *scratchGpu, void *scratchCpu, int length);
void freeGPUData(void *gpuData);
void matmul_cublas(float* xout, float* x, float* w, float* bias, float *d_B, float *d_C, int n, int d);
void matmul_cublas_qkv(float* c, float* a, float* b, float *d_A, float *d_B, float *d_C, int n, int d);