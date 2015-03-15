#include <helper_cuda.h>
#include <cufft.h>
#include <Ocean.h>
#include <glm/glm.hpp>
#include <thrust/complex.h>
#include <curand.h>


__global__ void gerstner(glm::vec3 *h_heightPointer,glm::vec3* h_normalPointer, wave* h_wavesPointer,float time, int _res, int _numWaves){

    // For all waves calucalte the height of the wave at (u, v)

    float u = float((threadIdx.x - (_res * floor(double(threadIdx.x / _res)))));
    float v = (float)((blockIdx.x * (1024.0/(float)_res)) + ceil(double(threadIdx.x / _res)) );

    // Find the positions previosuly in the buffer
    float xPrev = h_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x].x;
    float zPrev = h_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x].z;

    float waveX = 0;
    float waveY = 0;
    float waveZ  = 0;
    float waveL = 0;
    float waveR = 0;
    float waveU =0;
    float waveD= 0;

    // GPU GEMS ------------------------------------------------------------------------------------------------------------------------------------------------

//    for (int i=0; i<_numWaves; i++){
//        // Calcualte the x position
//        waveX += float((h_wavesPointer[i].Q * h_wavesPointer[i].A) * h_wavesPointer[i].D.x * cos(double(h_wavesPointer[i].W * glm::dot(h_wavesPointer[i].D, glm::vec2(u, v)) + h_wavesPointer[i].phaseConstant * time)));
//        // Calculate the y position
//        waveZ  += float( (h_wavesPointer[i].Q * h_wavesPointer[i].A) * h_wavesPointer[i].D.y * cos( double( h_wavesPointer[i].W * glm::dot( h_wavesPointer[i].D, glm::vec2(u, v)) + h_wavesPointer[i].phaseConstant * time  ) ) );
//        // Calculate the z position
//        waveY += float(h_wavesPointer[i].A * sin(double(h_wavesPointer[i].W * glm::dot(h_wavesPointer[i].D, glm::vec2(u, v)) + h_wavesPointer[i].phaseConstant * time)));
//        // Calculate the normals
//        waveL += float(h_wavesPointer[i].A * sin(double(h_wavesPointer[i].W * glm::dot(h_wavesPointer[i].D, glm::vec2(u-1.0, v)) + h_wavesPointer[i].phaseConstant * time)));
//        waveR += float(h_wavesPointer[i].A * sin(double(h_wavesPointer[i].W * glm::dot(h_wavesPointer[i].D, glm::vec2(u+1.0, v)) + h_wavesPointer[i].phaseConstant * time)));
//        waveU += float(h_wavesPointer[i].A * sin(double(h_wavesPointer[i].W * glm::dot(h_wavesPointer[i].D, glm::vec2(u, v-1.0)) + h_wavesPointer[i].phaseConstant * time)));
//        waveD += float(h_wavesPointer[i].A * sin(double(h_wavesPointer[i].W * glm::dot(h_wavesPointer[i].D, glm::vec2(u, v+1.0)) + h_wavesPointer[i].phaseConstant * time)));

//    }

//    h_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x].x = xPrev + waveX;
//    h_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x].y = waveY;
//    h_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x].z = zPrev + waveZ;

    // Tessendof
    for (int i=0; i<_numWaves; i++){
        // Calcualte the x position
        float w = sqrt(double(9.81 * h_wavesPointer[i].D.length()));
        glm::vec2 dir = glm::normalize(h_wavesPointer[i].D);
        waveX += float(dir.x * float(h_wavesPointer[i].A * sin(glm::dot(h_wavesPointer[i].D, glm::vec2(u, v)) - w * time + h_wavesPointer[i].phaseConstant)));
        waveZ += float(dir.y * float(h_wavesPointer[i].A * sin(glm::dot(h_wavesPointer[i].D, glm::vec2(u, v)) - w * time + h_wavesPointer[i].phaseConstant)));
        // Calculate the y position
        waveY += float(h_wavesPointer[i].A * cos(double(glm::dot(h_wavesPointer[i].D, glm::vec2(u, v)) - w * time + h_wavesPointer[i].phaseConstant)));
        // Calculate the normals
        waveL += float(h_wavesPointer[i].A * cos(double(glm::dot(h_wavesPointer[i].D, glm::vec2(u-1.0, v)) - w * time + h_wavesPointer[i].phaseConstant)));
        waveR += float(h_wavesPointer[i].A * cos(double(glm::dot(h_wavesPointer[i].D, glm::vec2(u+1.0, v)) - w * time + h_wavesPointer[i].phaseConstant)));
        waveU += float(h_wavesPointer[i].A * cos(double(glm::dot(h_wavesPointer[i].D, glm::vec2(u, v-1.0)) - w * time + h_wavesPointer[i].phaseConstant)));
        waveD += float(h_wavesPointer[i].A * cos(double(glm::dot(h_wavesPointer[i].D, glm::vec2(u, v+1.0)) - w * time + h_wavesPointer[i].phaseConstant)));
    }

    h_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x].x = xPrev - waveX;
    h_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x].z = zPrev - waveZ;
    h_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x].y = waveY*50;

    glm::vec3 N;
    N.x = waveL*50 - waveR*50;
    N.y = waveD*50 - waveU*50;
    N.z = 2.0;
    N.x = N.x /sqrt(N.x*N.x + N.y*N.y + N.z * N.z);
    N.y = N.y /sqrt(N.x*N.x + N.y*N.y + N.z * N.z);
    N.z = N.z /sqrt(N.x*N.x + N.y*N.y + N.z * N.z);

    h_normalPointer[(blockIdx.x * blockDim.x) + threadIdx.x] = N;
}

__global__ void FFTwaves(glm::vec2* h_h0Pointer, float2* h_htPointer, float time, int _res){
    float g = 9.81;

    glm::vec2 k;
    k.x = float((threadIdx.x - (_res * floor(double(threadIdx.x / _res)))) - (_res/2.0));
    k.y = float(((blockIdx.x * (blockDim.x/(float)_res)) + ceil(double(threadIdx.x / _res))) - (_res/2.0));
    float kLen = sqrt(double(k.x*k.x + k.y*k.y));

    float w = sqrt(double(g * kLen));

    // complexExp hold the complex exponential where the x value stores the real part and the y value stores the imaginary part
    glm::vec2 complexExp;
    complexExp.x = sin(w * time);
    complexExp.y = cos(w * time);
    if(complexExp != complexExp){
        printf("comEx: %f ,compExy: %f\n", complexExp.x, complexExp.y);

    }

    glm::vec2 complexExpConjugate;
    complexExpConjugate.x = complexExp.x;
    complexExpConjugate.y = -complexExp.y;

    glm::vec2 h0 = h_h0Pointer[(blockIdx.x * blockDim.x) + threadIdx.x];
    if(h0 != h0){
        printf("h0x: %f ,h0y: %f\n", h0.x, h0.y);
    }
    int blockNum =(( _res * _res )/ blockDim.x) - 1;

    glm::vec2 h0conjugate = h_h0Pointer[((blockNum - blockIdx.x) * blockDim.x) + ((blockDim.x - 1) - threadIdx.x)];
    if (h0conjugate != h0conjugate){
        printf("hconj0x: %f ,h0conjy: %f\n", h0conjugate.x, h0conjugate.y);
    }

    // Swap the imaginary parts sign
    h0conjugate.y = -h0conjugate.y;

    glm::vec2 h;
    h.x = (h0.x * complexExp.x - h0.y * complexExp.y);
    h.y = (h0.x * complexExp.x + h0.y * complexExp.y);
    if (h != h){
        printf("NAN h\n");

    }

    glm::vec2 hStar;
    hStar.x = (h0conjugate.x * complexExpConjugate.x - h0conjugate.y * complexExpConjugate.y) ;
    hStar.y = (h0conjugate.x * complexExpConjugate.x - h0conjugate.y * complexExpConjugate.y) ;
    if (hStar != hStar){
        printf("NAN hStar\n");
    }

    glm::vec2 hTilde = h + hStar;
    if (hTilde != hTilde){
        printf("NAN hTilde\n");
    }
    else{
        h_htPointer[(blockIdx.x * blockDim.x) + threadIdx.x].x = hTilde.x;
        h_htPointer[(blockIdx.x * blockDim.x) + threadIdx.x].y = hTilde.y;
    }
}

__global__ void height(glm::vec3* height, float2* ht, glm::vec3* _normal,glm::vec3* _colour, float2* _ht, int _res, float _scale, glm::vec3 _cameraPos){
    int u = int((threadIdx.x - (_res * floor(double(threadIdx.x / _res)))));
    int v = (int)((blockIdx.x * (blockDim.x/(float)_res)) + ceil(double(threadIdx.x / _res)) );
    float sign = 1.0;
    if ((u+v) % 2 != 0){
        sign = -1.0;
    }
    float htval = (ht[(blockIdx.x * blockDim.x) + threadIdx.x].x/_scale) * sign;
    height[(blockIdx.x * blockDim.x) + threadIdx.x].y = htval ;
//    height[(blockIdx.x * blockDim.x) + threadIdx.x].x = (_ht[(blockIdx.x * blockDim.x)+ threadIdx.x].x) * sign;
//    printf("Displace %f\n",  (_ht[(blockIdx.x * blockDim.x)+ threadIdx.y].x /_scale)* sign);

//    if ( height[(blockIdx.x * blockDim.x) + threadIdx.x].y > 16.0 ){
//       printf("Heigth: %f\n", height[(blockIdx.x * blockDim.x) + threadIdx.x].y);
//    }

    glm::vec3 norm;
    float nL, nR, nU, nD;
    if (((blockIdx.x * blockDim.x) + threadIdx.x) >=1)
        nL = (ht[((blockIdx.x * blockDim.x) + threadIdx.x) - 1].x/_scale) * sign;
    else
        nL = 0.0;
    if (((blockIdx.x * blockDim.x) + threadIdx.x) <= 65535)
        nR = (ht[((blockIdx.x * blockDim.x) + threadIdx.x) + 1].x/_scale) * sign;
    else
        nR = 0.0;
    if (((blockIdx.x * blockDim.x) + threadIdx.x) >= 256)
        nU = (ht[((blockIdx.x * blockDim.x) + threadIdx.x) - 256].x/_scale) * sign;
    else
        nU = 0.0;
    if (((blockIdx.x * blockDim.x) + threadIdx.x) <= 65279)
        nD = (ht[((blockIdx.x * blockDim.x) + threadIdx.x) + 256].x/_scale) * sign;
    else
        nD = 0.0;
    norm.x = nL - nR;
    norm.y = nD - nU;
    norm.z = 2.0;
    norm.x = norm.x /sqrt(norm.x*norm.x + norm.y*norm.y + norm.z * norm.z);
    norm.y = norm.y /sqrt(norm.x*norm.x + norm.y*norm.y + norm.z * norm.z);
    norm.z = norm.z /sqrt(norm.x*norm.x + norm.y*norm.y + norm.z * norm.z);

    _normal[(blockIdx.x * blockDim.x) + threadIdx.x] = norm;
}

__global__ void choppiness(float2* ht, int _res){
    float2 k;
    k.x = float((threadIdx.x - (_res * floor(double(threadIdx.x / _res)))) - (_res/2.0));
    k.y = float(((blockIdx.x * (blockDim.x/(float)_res)) + ceil(double(threadIdx.x / _res))) - (_res/2.0));
    float kLen = sqrt(double(k.x*k.x + k.y*k.y));
    float2 prev = ht[(blockIdx.x * blockDim.x) + threadIdx.x];
    float Kx = k.x / kLen;
    if (kLen == 0.0){
        Kx = 0.0;
    }
    ht[(blockIdx.x * blockDim.x) + threadIdx.x].x = prev.x * Kx;
}

void updateFFT(glm::vec2 *_h0, float2 *_ht, float _time, int _res){
    int numBlocks =( _res * _res )/ 1024;
    FFTwaves<<<numBlocks, 1024>>>(_h0, _ht, _time, _res);
}
void updateGerstner(glm::vec3 *_point, glm::vec3 *_point2, wave *_waves, float _time, int _res, int _numWaves){
    int numBlocks =( _res * _res )/ 1024;
    gerstner<<<numBlocks, 1024>>>(_point, _point2,_waves,  _time, _res, _numWaves);
}

void updateHeight(glm::vec3* _heightPointer, float2* _position, glm::vec3* _normal, glm::vec3* _colour,float2* _ht, int _res, float _scale, glm::vec3 _cameraPos){
    int numBlocks =( _res * _res )/ 1024;
    height<<<numBlocks, 1024>>>(_heightPointer, _position, _normal, _colour, _ht, _res, _scale, _cameraPos);
}

void addChoppiness(float2* ht, int _res){
    int numBlocks =( _res * _res )/ 1024;
    choppiness<<<numBlocks, 1024>>>(ht, _res);
}



