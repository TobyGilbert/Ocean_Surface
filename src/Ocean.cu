#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <Ocean.h>

const unsigned int res = 32;

__global__ void fillKernel(float *array){
    array[threadIdx.x] = threadIdx.x * 0.5;
}

__global__ void updateHeight(float *h_heightPointer, float etime){
//    // blck indx + floor(thrdindx /256)
//    //  floor(thread idx /256)
//    float u = float(threadIdx.x) / float(res);
//    float v = float(threadIdx.y) / float(res);
//    u = u*2.0f - 1.0f;
//    v = v*2.0f - 1.0f;

//    // calculate simple sine wave pattern
//    float freq = 6.0f;
//    float w = sin(u*freq + etime) * cos(v*freq + etime) * 0.25f;

//    // Block index * thread index
//    // 4 lines for every block
//    // Block indx * 256 * 4 + thread indx
//    // write output vertex
//    h_heightPointer[threadIdx.x*res+threadIdx.y] = w*2.0;

    float u = float((threadIdx.x - (floor((double)threadIdx.x / 256.0)*256.0))/256.0);
    float v = float((blockIdx.x + floor((double)threadIdx.x/256.0))/256.0);

    float freq = 6.0f;
    float w = sin(u*freq + etime) * cos(v*freq + etime) * 0.25f;

    h_heightPointer[(blockIdx.x * 256 * 4) + threadIdx.x] =w*100;

}

void fillGPUArray(float *array, int count){
    fillKernel<<<1, count>>>(array);
}

cudaGraphicsResource_t resourceHeight;

void registerGLBuffer(GLuint _GLBuffer){
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&resourceHeight, _GLBuffer, cudaGraphicsRegisterFlagsWriteDiscard));
}

void updateHeight(double _time){
    // Map the graphics resource
    cudaGraphicsMapResources(1, &resourceHeight);

    // Get a pointer to the buffer
    float* mapPointer;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&mapPointer, &numBytes, resourceHeight);

    // Need 64 blocks of 1024 threads
    dim3 d(res, res, 1);
    updateHeight<<<64, 1024>>>(mapPointer, (float)_time);

    cudaGraphicsUnmapResources(1, &resourceHeight);
}
