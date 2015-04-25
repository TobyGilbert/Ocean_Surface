#include <iostream>
#include <string>
#ifdef DARWIN
#include <OpenGL/gl3.h>
#else
#include <GL/glew.h>
#include <GL/gl.h>
#endif

class Model{
public:
   Model(std::string _path);
   Model();
   ~Model();
   //void loadModel(std::string _path);
   GLuint getVAO(){return m_VAO;}
   void loadCube();
   GLuint getNumVerts(){return m_numVerts;}
private:
   GLuint m_vertexVBO;
   GLuint m_normalsVBO;
   GLuint m_textureVBO;
   GLuint m_tangentsVBO;
   GLuint m_cubeVBO;
   GLuint m_VAO;
   GLuint m_numVerts;
   void loadModel(std::string _path);

};
