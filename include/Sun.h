#ifndef SUN_H
#define SUN_H
#include "ModelLoader.h"
#include "ShaderProgram.h"
#include <glm/glm.hpp>

class Sun{
public:
    Sun();
    ~Sun();
    void initialise();
    void loadMatricesToShader(glm::mat4 _modelMatrix, glm::mat4 _viewMatrix, glm::mat4 _projectionMatrix);
    void render();
private:
    void createShader();
    ModelLoader *m_model;
    ShaderProgram *m_shaderProgram;
    Shader *m_vertShader;
    Shader *m_fragShader;

};


#endif
