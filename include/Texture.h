/** @addtogroup OceanFFTStandAlone */
/*@{*/

#ifndef TEXTURE_H
#define TEXTURE_H
//----------------------------------------------------------------------------------------------------------------------
#include <iostream>
#ifdef DARWIN
    #include <GLFW/glfw3.h>
#else
    #include <GL/glew.h>
    #include <GL/gl.h>
#endif
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
//----------------------------------------------------------------------------------------------------------------------
/// @brief A class for loading textures from files to OpenGL textures
/// @author Toby Gilbert
//----------------------------------------------------------------------------------------------------------------------
class Texture{
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Constructor
    //----------------------------------------------------------------------------------------------------------------------
    Texture(std::string _path);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Destructor
    //----------------------------------------------------------------------------------------------------------------------
    ~Texture();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets the active texture to bind to
    /// @param _unit the active texture to bind to
    //----------------------------------------------------------------------------------------------------------------------
    void bind(GLuint _unit);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Retuns the texture handle id
    /// @return the texture handle id
    //----------------------------------------------------------------------------------------------------------------------
    GLuint getTextureID();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets OpenGL texture parameters
    /// @param _pname the parameter name
    /// @param _param the paramter argument
    //----------------------------------------------------------------------------------------------------------------------
    void setParamater(GLenum _pname, GLenum _param);
    //----------------------------------------------------------------------------------------------------------------------
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The texture id
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_textureID;
    //----------------------------------------------------------------------------------------------------------------------
};
//----------------------------------------------------------------------------------------------------------------------
#endif
/*@}*/
