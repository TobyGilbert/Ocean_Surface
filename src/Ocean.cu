// ----------------------------------------------------------------------------------------------------------------------------------------
/// @author Toby Gilbert
// ----------------------------------------------------------------------------------------------------------------------------------------
#include "Ocean.h"
#include <helper_cuda.h>
#include <cufft.h>
#include <glm/glm.hpp>
#include <complex.h>
//#include <thrust/complex.h>
#include <curand.h>
#include <helper_functions.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <helper_math.h>
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief Uses Gerstner's equation for generating waves on a regular grid
/// @param d_heightPointer An OpenGL buffer for storing the position information for the vertices on the grid
/// @param d_normalPointer An OpenGL buffer for storing vertex normals
/// @param d_wavesPointer An OpenGL buffer for storing wave information
/// @param _time The current simulatation time
/// @param _res The resolution of the simulation grid
/// @param _numWaves The number of waves in the simulation
// ----------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------
__global__ void gerstner(glm::vec3 *d_heightPointer,glm::vec3* d_normalPointer, wave* d_wavesPointer,float _time, int _res, int _numWaves){

    // For all waves calucalte the height of the wave at (u, v)
    float u = float((threadIdx.x - (_res * floor(double(threadIdx.x / _res)))));
    float v = (float)((blockIdx.x * (1024.0/(float)_res)) + ceil(double(threadIdx.x / _res)) );

    // Find the positions previosuly in the buffer
    float xPrev = d_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x].x;
    float zPrev = d_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x].z;

    float waveX = 0;
    float waveY = 0;
    float waveZ = 0;
    float waveL = 0;
    float waveR = 0;
    float waveU = 0;
    float waveD = 0;

    // Tessendof
    for (int i=0; i<_numWaves; i++){
        // Calcualte the x position
        float w = sqrt(double(9.81 * d_wavesPointer[i].D.length()));
        glm::vec2 dir = glm::normalize(d_wavesPointer[i].D);
        waveX += float(dir.x * float(d_wavesPointer[i].A * sin(glm::dot(d_wavesPointer[i].D, glm::vec2(u, v)) - w * _time + d_wavesPointer[i].phaseConstant)));
        waveZ += float(dir.y * float(d_wavesPointer[i].A * sin(glm::dot(d_wavesPointer[i].D, glm::vec2(u, v)) - w * _time + d_wavesPointer[i].phaseConstant)));
        // Calculate the y position
        waveY += float(d_wavesPointer[i].A * cos(double(glm::dot(d_wavesPointer[i].D, glm::vec2(u, v)) - w * _time + d_wavesPointer[i].phaseConstant)));
        // Calculate the normals
        waveL += float(d_wavesPointer[i].A * cos(double(glm::dot(d_wavesPointer[i].D, glm::vec2(u-1.0, v)) - w * _time + d_wavesPointer[i].phaseConstant)));
        waveR += float(d_wavesPointer[i].A * cos(double(glm::dot(d_wavesPointer[i].D, glm::vec2(u+1.0, v)) - w * _time + d_wavesPointer[i].phaseConstant)));
        waveU += float(d_wavesPointer[i].A * cos(double(glm::dot(d_wavesPointer[i].D, glm::vec2(u, v-1.0)) - w * _time + d_wavesPointer[i].phaseConstant)));
        waveD += float(d_wavesPointer[i].A * cos(double(glm::dot(d_wavesPointer[i].D, glm::vec2(u, v+1.0)) - w * _time + d_wavesPointer[i].phaseConstant)));
    }

    d_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x].x = xPrev - waveX;
    d_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x].z = zPrev - waveZ;
    d_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x].y = waveY*50;

    glm::vec3 N;
    N.x = waveL*50 - waveR*50;
    N.y = 1.0;
    N.z = waveD*50 - waveU*50;
    N.x = N.x /sqrt(N.x*N.x + N.y*N.y + N.z * N.z);
    N.y = N.y /sqrt(N.x*N.x + N.y*N.y + N.z * N.z);
    N.z = N.z /sqrt(N.x*N.x + N.y*N.y + N.z * N.z);

    d_normalPointer[(blockIdx.x * blockDim.x) + threadIdx.x] = N;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief Given a time you can create a field of frequency amplitudes
/// @param d_h0Pointer An OpenGL buffer which stores a set of amplitudes and phases at time zero
/// @param d_htPointer An OpenGL buffer for outputting the frequency amplitude field
/// @param _time The current simulation time
/// @param _res The simulation resolution
// ----------------------------------------------------------------------------------------------------------------------------------------
__global__ void frequencyDomain(float2* d_h0Pointer, float2* d_htPointer, float _time, int _res){
    // A constant for the accelleration due to gravity
    const float g = 9.81;

    // A 2D vector to represent a position on the grid with constraits -(_res/2) <= k < (_res/2)
    float2 k;
    k.x = float((threadIdx.x - (_res * floor(double(threadIdx.x / _res)))) - (_res/2));
//    printf("k.x %f\n", k.x);
    k.y = float(((blockIdx.x * (blockDim.x/_res)) + ceil(double(threadIdx.x / _res))) - (_res/2));
//    printf("k.y %f\n", k.y);
    float kLen = sqrt(double(k.x*k.x + k.y*k.y));

    // Calculate the wave frequency
    float w = sqrt(double(g * kLen));

    // complexExp holds the complex exponential where the x value stores the real part and the y value stores the imaginary part
    float2 complexExp;
    complexExp.x = sin(w * _time);
    complexExp.y = cos(w * _time);

    float2 complexExpConjugate;
    complexExpConjugate.x = complexExp.x;
    complexExpConjugate.y = -complexExp.y;

    int blockNum =(( _res * _res )/ blockDim.x) - 1;

    float2 h0 = d_h0Pointer[(blockIdx.x * blockDim.x) + threadIdx.x];
    float2 h0conjugate = d_h0Pointer[((blockNum - blockIdx.x) * blockDim.x) + ((blockDim.x - 1) - threadIdx.x)];

    // Swap the imaginary parts sign
    h0conjugate.y = -h0conjugate.y;

    // Equation 26 of Tessendorf's paper h(k,t) = h0(k)exp{iw(k)t} + ~h0(-k)exp{-iw(k)t}
    float2 h;
    h.x = (h0.x * complexExp.x - h0.y * complexExp.y);
    h.y = (h0.x * complexExp.x + h0.y * complexExp.y);

    float2 hStar;
    hStar.x = (h0conjugate.x * complexExpConjugate.x - h0conjugate.y * complexExpConjugate.y) ;
    hStar.y = (h0conjugate.x * complexExpConjugate.x - h0conjugate.y * complexExpConjugate.y) ;

    // Output h(k,t) term to d_htPointer buffer which represents a set of points in the frequency domain
    float2 hTilde;
    hTilde.x= h.x + hStar.x;
    hTilde.y = h.y + hStar.y;

//    printf("hTilde %f\n", hTilde.x);

    d_htPointer[(blockIdx.x * blockDim.x) + threadIdx.x].x = hTilde.x;
    d_htPointer[(blockIdx.x * blockDim.x) + threadIdx.x].y = hTilde.y;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief Once inverse FFT has been performed points in the frequency domain are converted to the spatial domain
/// and can be used to update the heights
/// @param d_position An OpenGL buffer for storing the current positions of the vertices in the grid
/// @param d_height An OpenGL buffer which holds the new heights of grid positions
/// @param d_normal An OpenGL buffer which holds the normals
/// @param d_xDisplacement An OpenGL buffer for storing the displacment in the x axis
/// @param _res The resolution of the grid
/// @param _scale Scales the amplitude of the waves
// ----------------------------------------------------------------------------------------------------------------------------------------
__global__ void height(float3* d_position,  float2* d_height, float2* d_chopX, float2* d_chopZ, float _choppiness, int _res, float _scale){
    // A vertex on the grid
    int u = int(threadIdx.x - (_res * floor(double(threadIdx.x / _res))));
    int v = int((blockIdx.x * (blockDim.x/(float)_res)) + ceil(double(threadIdx.x / _res)));

    // Sign correction - Unsure why this is needed
    float sign = 1.0;
    if ((u+v) % 2 != 0){
        sign = -1.0;
    }

    // Update the heights of the vertices
    float prevX = d_position[(blockIdx.x * blockDim.x) + threadIdx.x].x;
    float prevZ = d_position[(blockIdx.x * blockDim.x) + threadIdx.x].z;
//    float xDisp = /*_choppiness */ d_chopX[(blockIdx.x * blockDim.x) + threadIdx.x].x;//  /_scale) * sign;
   // float zDisp = _choppiness * (d_chopZ[(blockIdx.x * blockDim.x) + threadIdx.x].x  /_scale) * sign;
//    float xDisp = 0.0;
    float zDisp = 0.0;
    float height =  ((d_height[(blockIdx.x * blockDim.x) + threadIdx.x].x / _scale) * sign ) / 255.0f;
    float3 newPos = make_float3(prevX, height, prevZ);

    printf("new Pos %f %f %f \n", newPos.x, newPos.y, newPos.z);

    d_position[(blockIdx.x * blockDim.x) + threadIdx.x] = newPos;
}

__global__ void calculateNormals(float3* d_position, float3* d_normals, int _res, float _scale){

    float3 norm = make_float3(0.0, 0.0, 0.0);
    float3 posL, posR, posD, posU;
    // TODO remove branching conditions
    if (((blockIdx.x * blockDim.x) + threadIdx.x) >= 1){
        posL = (d_position[((blockIdx.x * blockDim.x) + threadIdx.x) - 1]);
    }
    else{
        posL = make_float3(0.0, 0.0, 0.0);
    }
    if (((blockIdx.x * blockDim.x) + threadIdx.x) <=(_res*_res)-2){
        posR = (d_position[((blockIdx.x * blockDim.x) + threadIdx.x) + 1]);
    }
    else{
        posR = make_float3(0.0, 0.0, 0.0);
    }
    if (((blockIdx.x * blockDim.x) + threadIdx.x) >= _res){
        posU = (d_position[((blockIdx.x * blockDim.x) + threadIdx.x) - _res]);
    }
    else{
        posU = make_float3(0.0, 0.0, 0.0);
    }
    if (((blockIdx.x * blockDim.x) + threadIdx.x) <= (_res*_res)-_res-1){
        posD = (d_position[((blockIdx.x * blockDim.x) + threadIdx.x) + _res]);
    }
    else{
        posD = make_float3(0.0, 0.0, 0.0);
    }


    float3 leftVec, rightVec, topVec, bottomVec;
    float3 centerVec = d_position[((blockIdx.x * blockDim.x) + threadIdx.x)];
    leftVec =  posL - centerVec;
    leftVec.y *= 100.0;
    rightVec = posR - centerVec;
    rightVec.y *= 100.0;
    topVec = posU - centerVec;
    topVec.y *= 100.0;
    bottomVec =  posD - centerVec;
    bottomVec.y *= 100.0;
//    printf("leftVec %f, %f, %f\n ", leftVec.x, leftVec.y, leftVec.z);
//    printf("rightVec %f, %f, %f\n ", rightVec.x, rightVec.y, rightVec.z);
//    printf("topVec %f, %f, %f\n ", topVec.x, topVec.y, topVec.z);
//    printf("bottomVec %f, %f, %f\n ", bottomVec.x, bottomVec.y, bottomVec.z);


    float3 tmpNorm1 = normalize(cross(leftVec, topVec));
    float3 tmpNorm2 = normalize(cross(topVec, rightVec));
    float3 tmpNorm3 = normalize(cross(rightVec, bottomVec));
    float3 tmpNorm4 = normalize(cross(bottomVec, leftVec));

    tmpNorm1.y = fabs(tmpNorm1.y);
    tmpNorm2.y = fabs(tmpNorm2.y);
    tmpNorm3.y = fabs(tmpNorm3.y);
    tmpNorm4.y = fabs(tmpNorm4.y);
    norm = normalize((tmpNorm1 + tmpNorm2 + tmpNorm3 + tmpNorm4));

    // Update the normals buffer
    d_normals[(blockIdx.x * blockDim.x) + threadIdx.x] = norm;
}

// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief Create x displacement in in the frequency domain
/// @param
/// @param d_xDisplacement An OpenGL buffer to store the x displacement in the frequency domain
/// @param d_zDisplacement An OpenGL buffer to store the z displacement in the frequency domain
/// @param _res The resolution of the grid
// ----------------------------------------------------------------------------------------------------------------------------------------
__global__ void choppiness(float2* d_Ht, float2* d_chopX, float2* d_chopZ, float2 _windSpeed){
    // k - A position on the grid
    float2 k;
    k.x = _windSpeed.x;
    k.y = _windSpeed.y;

    float kLen = sqrt(double(k.x*k.x + k.y*k.y));

    float Kx = k.x / kLen;
    float Kz = k.y / kLen;

    if (kLen == 0.0){
        Kx = 0.0;
        Kz = 0.0;
    }

    d_chopX[(blockIdx.x * blockDim.x) + threadIdx.x].x = 0.0;
    d_chopX[(blockIdx.x * blockDim.x) + threadIdx.x].y = d_Ht[(blockIdx.x * blockDim.x) + threadIdx.x].y * -Kx;

    d_chopZ[(blockIdx.x * blockDim.x) + threadIdx.x].x = 0.0;
    d_chopZ[(blockIdx.x * blockDim.x) + threadIdx.x].y = d_Ht[(blockIdx.x * blockDim.x) + threadIdx.x].y * -Kz;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void updateFrequencyDomain(float2 *d_h0, float2 *d_ht, float _time, int _res){
    int numBlocks =( _res * _res )/ 1024;
    frequencyDomain<<<numBlocks, 1024>>>(d_h0, d_ht, _time, _res);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void updateGerstner(glm::vec3 *d_heightPointer,glm::vec3* d_normalPointer, wave *d_waves, float _time, int _res, int _numWaves){
    int numBlocks =( _res * _res )/ 1024;
    gerstner<<<numBlocks, 1024>>>(d_heightPointer, d_normalPointer, d_waves,  _time, _res, _numWaves);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void updateHeight(float3* d_position, float3* d_norms, float2* d_height, float2* d_chopX, float2* d_chopZ, float _choppiness, int _res, float _scale){
    int numBlocks =( _res * _res )/ 1024;
    height<<<numBlocks, 1024>>>(d_position, d_height, d_chopX, d_chopZ, _choppiness,  _res, _scale);

    cudaThreadSynchronize();

    calculateNormals<<<numBlocks, 1024>>>(d_position, d_norms, _res, _scale);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void addChoppiness(float2* d_Heights, float2* d_chopX, float2* d_chopZ, int _res, float2 _windSpeed){
    int numBlocks =( _res * _res )/ 1024;
    choppiness<<<numBlocks, 1024>>>(d_Heights, d_chopX, d_chopZ, _windSpeed);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
