#include <iostream>
#include <GLFW/glfw3.h>
#include <IL/il.h>
#include <glm/glm.hpp>

class Texture{
public:
   Texture(std::string _path);
   ~Texture();
   void bind(GLuint _unit);
   GLuint getTextureID();
   void setParamater(GLenum _pname, GLenum _param);
private:
   GLuint m_textureID;

};
