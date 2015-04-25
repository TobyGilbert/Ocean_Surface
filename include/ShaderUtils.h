#include <iostream>
#ifdef DARWIN
#include <GLFW/glfw3.h>
#else
#include <GL/glew.h>
#include <GL/gl.h>
#endif
#include <fstream>
#include <streambuf>
#include <string>

class shaderUtils{
public:
   static GLuint createShaderFromFile(const GLchar* path, GLenum shaderType);
};
