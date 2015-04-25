#ifndef __SHADER_H_
#define __SHADER_H_
//-------------------------------------------------------------------------------------------------------------------------
#ifdef DARWIN
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
    #include <GL/gl.h>
#endif
//-------------------------------------------------------------------------------------------------------------------------
#include <iostream>
//#include <IL/il.h>
#include <glm/glm.hpp>
#include <string>
//-------------------------------------------------------------------------------------------------------------------------
class Shader{
public:
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief ctor
   //----------------------------------------------------------------------------------------------------------------------
   Shader(std::string _path, GLenum _type);
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief dtor
   //----------------------------------------------------------------------------------------------------------------------
   virtual ~Shader();
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief Returns the shader ID
   //----------------------------------------------------------------------------------------------------------------------
   GLuint getShaderID();
   //----------------------------------------------------------------------------------------------------------------------
private:
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief Shader ID
   //----------------------------------------------------------------------------------------------------------------------
   GLuint m_shaderID;
   //----------------------------------------------------------------------------------------------------------------------
};
//-------------------------------------------------------------------------------------------------------------------------
#endif
