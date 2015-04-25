#include <iostream>
#ifdef DARWIN
    #include <GLFW/glfw3.h>
#else
    #include <GL/glew.h>
    #include <GL/gl.h>
#endif
//#include <IL/il.h>

class TextureUtils{
public:
   static GLuint createTexture(const GLchar* path);
};
