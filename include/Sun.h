#ifndef SUN_H
#define SUN_H
#include <glm/glm.hpp>
#include "ModelLoader.h"
#include "ShaderProgram.h"

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
