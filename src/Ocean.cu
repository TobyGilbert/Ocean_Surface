#include <helper_cuda.h>
#include <Ocean.h>
#include <glm/glm.hpp>

const unsigned int res = 256;

__global__ void updateHeight(float *h_heightPointer,glm::vec3* h_normalPointer, float etime){

    // For all waves
    float pi = 3.14;
    float W = 0.4;
    float A = 10.0;
    float Q = 0.4;
    glm::vec2 D = glm::vec2(0.5, 0.5);
    float S = 1.0;
    float L = 0.6;
    float phaseConstant = S * (pi / L);

    float u = float((threadIdx.x - (res * floor(double(threadIdx.x / res))))/(float)res);
    float v = (float)((blockIdx.x * (1024.0/(float)res)) + ceil(double(threadIdx.x / res)) )/ (float)res;

//    yPos += (float) ((steepness[i] * amplitude[i]) * direction[i].y * Math.cos(w[i] * (direction[i].dot(position[i])) + phase_const[i] * time));


    float wave  = float( (Q * A) * D.y * cos( double( W * glm::dot( D, glm::vec2(u*256.0, v*256.0)) + phaseConstant * etime  ) ) );

    // Calculate the normals
    float waveL = float( (Q * A) * D.y * cos( double( W * glm::dot( D, glm::vec2((u*256.0)-1.0, v*256.0)) + phaseConstant * etime  ) ) );
    float waveR = float( (Q * A) * D.y * cos( double( W * glm::dot( D, glm::vec2((u*256.0)+1, v*256.0)) + phaseConstant * etime  ) ) );
    float waveU = float( (Q * A) * D.y * cos( double( W * glm::dot( D, glm::vec2(u*256.0, (v*256.0)-1.0)) + phaseConstant * etime  ) ) );
    float waveD = float( (Q * A) * D.y * cos( double( W * glm::dot( D, glm::vec2(u*256.0, (v*256.0)+1.0)) + phaseConstant * etime  ) ) );



//    float u = float((threadIdx.x - (res * floor(double(threadIdx.x / res))))/(float)res);
//    float v = (float)((blockIdx.x * (1024.0/(float)res)) + ceil(double(threadIdx.x / res)) )/ (float)res;
//    float freq = 10.0f;
//    float w = 100*(sin(u*freq + etime) * cos(v*freq + etime) * 0.25f);
//    float wL = 100* (sin((u-(1.0/res))*freq + etime) * cos(v*freq + etime) * 0.25f);
//    float wR = 100*(sin((u+(1.0/res))*freq + etime) * cos(v*freq + etime) * 0.25f);
//    float wU = 100*(sin(u*freq + etime) * cos((v-(1.0/res))*freq + etime) * 0.25f);
//    float wD = 100*(sin(u*freq + etime) * cos((v+(1.0/res))*freq + etime) * 0.25f);
    glm::vec3 N;
    N.x = waveL - waveR;
    N.y = waveD - waveU;
    N.z = 2.0;
    N.x = N.x /sqrt(N.x*N.x + N.y*N.y + N.z * N.z);
    N.y = N.y /sqrt(N.x*N.x + N.y*N.y + N.z * N.z);
    N.z = N.z /sqrt(N.x*N.x + N.y*N.y + N.z * N.z);

    float height = h_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x];

    h_heightPointer[(blockIdx.x * blockDim.x) + threadIdx.x] = height + (wave);
    h_normalPointer[(blockIdx.x * blockDim.x) + threadIdx.x] = N;
}

void updateGeometry(float *_point, glm::vec3 *_point2, float _time){
    int numBlocks =( res * res )/ 1024;
    updateHeight<<<numBlocks, 1024>>>(_point, _point2, _time);
}

