#ifndef OCEAN_H_
#define OCEAN_H_

void fillGPUArray(float *array, int count);
class Ocean{
public:
    Ocean();
    ~Ocean();
};

void registerGLBuffer(GLuint _GLBuffer);
void updateHeight(double _time);

#endif
