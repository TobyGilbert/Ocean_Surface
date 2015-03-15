#ifndef OCEAN_H_
#define OCEAN_H_
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>

struct wave{
    float W;
    float A;
    float Q;
    glm::vec2 D;
    float S;
    float L;
    float phaseConstant;
};

void registerHeightBuffer(GLuint _GLBuffer);
void registerNormalBuffer(GLuint _GLBuffer);
void updateHeight(glm::vec3* _heightPointer, float2* _position, glm::vec3 *_normal, glm::vec3 *_colour, float2* _ht, int _res, float _scale = 1000.0, glm::vec3 _cameraPos = glm::vec3(0.0, 20.0, 50.0));
void updateGerstner(glm::vec3 *_point, glm::vec3 *_point2, wave* _waves, float _time, int _res, int _numWaves);
void updateFFT(glm::vec2 *_h0, float2 *_ht, float _time, int _res);
void addChoppiness(float2* ht, int _res);
#endif
