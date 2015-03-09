#ifndef OCEAN_H_
#define OCEAN_H_
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>

void registerHeightBuffer(GLuint _GLBuffer);
void registerNormalBuffer(GLuint _GLBuffer);
void updateHeight(double _time);
void updateGeometry(float* _point, glm::vec3 *_point2, float _time);
#endif
