/** @addtogroup OceanFFTStandAlone */
/*@{*/

#ifndef SHADERUTILS_H
#define SHADERUTILS_H

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
//----------------------------------------------------------------------------------------------------------------------
/// @brief A helper class for creating shaders from txt files
//----------------------------------------------------------------------------------------------------------------------
class shaderUtils{
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Creates an OpenGL shader from a txt file
    /// @param path a path to the txt file
    /// @param shaderType an OpenGL enum eg. GL_VERTEX_SHADER, GL_FRAGMENT_SHADER etc.
    //----------------------------------------------------------------------------------------------------------------------
    static GLuint createShaderFromFile(const GLchar* path, GLenum shaderType);
    //----------------------------------------------------------------------------------------------------------------------
};

#endif
/*@}*/
