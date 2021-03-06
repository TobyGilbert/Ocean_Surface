/** @addtogroup OceanFFTStandAlone */
/*@{*/

#ifndef __SHADERPROGRAM_H_
#define __SHADERPROGRAM_H_
//----------------------------------------------------------------------------------------------------------------------
#ifdef DARWIN
    #include <GLFW/glfw3.h>
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
    #include <GL/gl.h>
#endif
//----------------------------------------------------------------------------------------------------------------------
#include "Shader.h"
//----------------------------------------------------------------------------------------------------------------------
#include <iostream>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <vector>
//----------------------------------------------------------------------------------------------------------------------
/// @brief A class for creating OpenGL shader program from shader objects
/// @author Toby Gilbert
//----------------------------------------------------------------------------------------------------------------------
class ShaderProgram{
public:
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief ctor
   //----------------------------------------------------------------------------------------------------------------------
   ShaderProgram();
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief dtor
   //----------------------------------------------------------------------------------------------------------------------
   virtual ~ShaderProgram();
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief Attaches shader the shader program
   //----------------------------------------------------------------------------------------------------------------------
   void attachShader(Shader* _shader);
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief Binds the fragColour to a location
   //----------------------------------------------------------------------------------------------------------------------
   void bindFragDataLocation(GLuint _colourAttatchment, std::string _name);
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief Links the shader program
   //----------------------------------------------------------------------------------------------------------------------
   void link();
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief Use the shader program
   //----------------------------------------------------------------------------------------------------------------------
   void use();
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief Returns the shader program ID
   //----------------------------------------------------------------------------------------------------------------------
   inline GLuint getProgramID(){return m_programID;}
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief Returns a shaders attributes locations
   //----------------------------------------------------------------------------------------------------------------------
   GLint getAttribLoc(std::string _name);
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief Returns a shader uniforms location
   //----------------------------------------------------------------------------------------------------------------------
   GLint getUniformLoc(std::string _name);
   //----------------------------------------------------------------------------------------------------------------------
private:
   //----------------------------------------------------------------------------------------------------------------------
   /// @brief Shader program ID
   //----------------------------------------------------------------------------------------------------------------------
   GLuint m_programID;
   //----------------------------------------------------------------------------------------------------------------------
};
//-------------------------------------------------------------------------------------------------------------------------
#endif
/*@}*/

