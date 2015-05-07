#include "Boat.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

using glm::rotate;
//----------------------------------------------------------------------------------------------------------------------
Boat::Boat(std::string _model, std::string _texture): m_angle(0.0), m_forwards(true), m_yTrans(0.0){
    createShader(_texture);
    initialise(_model);
}
//----------------------------------------------------------------------------------------------------------------------
Boat::~Boat(){
    delete m_shaderProgram;
    delete m_model;
}
//----------------------------------------------------------------------------------------------------------------------
void Boat::createShader(std::string _texture){
    m_shaderProgram = new ShaderProgram();
    m_vertShader = new Shader("shaders/PhongVert.glsl", GL_VERTEX_SHADER);
    m_fragShader = new Shader("shaders/PhongFrag.glsl", GL_FRAGMENT_SHADER);
    m_shaderProgram->attachShader(m_vertShader);
    m_shaderProgram->attachShader(m_fragShader);
    m_shaderProgram->bindFragDataLocation(0, "fragColour");
    m_shaderProgram->link();
    m_shaderProgram->use();

    delete m_vertShader;

    GLuint lightPosLoc = m_shaderProgram->getUniformLoc("light.position");
    GLuint lightIntLoc = m_shaderProgram->getUniformLoc("light.intensity");
    GLuint KdLoc = m_shaderProgram->getUniformLoc("Kd");
    GLuint KaLoc = m_shaderProgram->getUniformLoc("Ka");
    GLuint KsLoc = m_shaderProgram->getUniformLoc("Ks");
    GLuint shininessLoc = m_shaderProgram->getUniformLoc("shininess");
    GLuint texLoc = m_shaderProgram->getUniformLoc("tex");

    glUniform4f(lightPosLoc, 0.0, 20.0, -500.0, 1.0);
    glUniform3f(lightIntLoc, 1.0, 1.0, 1.0);
    glUniform3f(KdLoc, 0.7, 0.7, 0.7);
    glUniform3f(KaLoc, 0.3, 0.3, 0.3);
    glUniform3f(KsLoc, 0.7, 0.7, 0.7);
    glUniform1f(shininessLoc, 1000.0);
    glUniform1i(texLoc, 0);

    //-----------The same shader program using a clipping plane-----------

    m_shaderProgramClipped = new ShaderProgram();
    m_vertShaderClipped = new Shader("shaders/clippedPhongVert.glsl", GL_VERTEX_SHADER);
    m_shaderProgramClipped->attachShader(m_vertShaderClipped);
    m_shaderProgramClipped->attachShader(m_fragShader);
    m_shaderProgramClipped->bindFragDataLocation(0, "fragColour");
    m_shaderProgramClipped->link();
    m_shaderProgramClipped->use();

    delete m_vertShaderClipped;
    delete m_fragShader;

    lightPosLoc = m_shaderProgramClipped->getUniformLoc("light.position");
    lightIntLoc = m_shaderProgramClipped->getUniformLoc("light.intensity");
    KdLoc = m_shaderProgramClipped->getUniformLoc("Kd");
    KaLoc = m_shaderProgramClipped->getUniformLoc("Ka");
    KsLoc = m_shaderProgramClipped->getUniformLoc("Ks");
    shininessLoc = m_shaderProgramClipped->getUniformLoc("shininess");
    texLoc = m_shaderProgramClipped->getUniformLoc("woodTexture");

    glUniform4f(lightPosLoc, 0.0, 10.0, 50.0, 1.0);
    glUniform3f(lightIntLoc, 1.0, 1.0, 1.0);
    glUniform3f(KdLoc, 0.3, 0.3, 0.3);
    glUniform3f(KaLoc, 0.3, 0.3, 0.3);
    glUniform3f(KsLoc, 0.3, 0.3, 0.3);
    glUniform1f(shininessLoc, 100.0);
    glUniform1i(texLoc, 0);

    m_tex = new Texture(_texture);
    m_tex->bind(0);

}
//----------------------------------------------------------------------------------------------------------------------
void Boat::initialise(std::string _model){
    m_model = new ModelLoader((char*)_model.c_str());
}
//----------------------------------------------------------------------------------------------------------------------
void Boat::loadMatricesToShader(glm::mat4 _modelMatrix, glm::mat4 _viewMatrix, glm::mat4 _projectionMatrix){
    m_shaderProgram->use();
    GLuint modelViewLoc = m_shaderProgram->getUniformLoc("modelViewMatrix");
    GLuint normalLoc = m_shaderProgram->getUniformLoc("normalMatrix");
    GLuint modelViewProjectionLoc = m_shaderProgram->getUniformLoc("modelViewProjectionMatrix");

    // Rotate the model matrix to ad the effect of floating
    glm::mat4 modelMatrix = _modelMatrix;
    glm::mat4 rotationMatrix = glm::mat4(1.0);
    rotationMatrix[1][1] = cos(m_angle);
    rotationMatrix[1][2] = -sin(m_angle);
    rotationMatrix[2][1] = sin(m_angle);
    rotationMatrix[2][2] = cos(m_angle);

    modelMatrix *= rotationMatrix;
    modelMatrix[3][1] += m_yTrans;

    glm::mat4 modelView = _viewMatrix * modelMatrix;
    glm::mat3 normalMatrix = glm::inverseTranspose(glm::mat3(modelView));
    glm::mat4 modelViewProjection = _projectionMatrix * modelView;

    glUniformMatrix4fv(modelViewLoc, 1, GL_FALSE, glm::value_ptr(modelView));
    glUniformMatrix3fv(normalLoc, 1, GL_FALSE, glm::value_ptr(normalMatrix));
    glUniformMatrix4fv(modelViewProjectionLoc, 1, GL_FALSE, glm::value_ptr(modelViewProjection));
}

void Boat::loadMatricesToShaderClipped(glm::mat4 _modelMatrix, glm::mat4 _viewMatrix, glm::mat4 _projectionMatrix){
    m_shaderProgramClipped->use();
    GLuint modelLoc = m_shaderProgram->getUniformLoc("modelMatrix");
    GLuint modelViewLoc = m_shaderProgram->getUniformLoc("modelViewMatrix");
    GLuint normalLoc = m_shaderProgram->getUniformLoc("normalMatrix");
    GLuint modelViewProjectionLoc = m_shaderProgram->getUniformLoc("modelViewProjectionMatrix");

    // Rotate the model matrix to ad the effect of floating
    glm::mat4 modelMatrix = _modelMatrix;
    glm::mat4 rotationMatrix = glm::mat4(1.0);
    rotationMatrix[1][1] = cos(m_angle);
    rotationMatrix[1][2] = -sin(m_angle);
    rotationMatrix[2][1] = sin(m_angle);
    rotationMatrix[2][2] = cos(m_angle);

    modelMatrix *= rotationMatrix;
    modelMatrix[3][1] -= m_yTrans;

    glm::mat4 modelView = _viewMatrix * modelMatrix;
    glm::mat3 normalMatrix = glm::inverseTranspose(glm::mat3(modelView));
    glm::mat4 modelViewProjection = _projectionMatrix * modelView;

    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(modelViewLoc, 1, GL_FALSE, glm::value_ptr(modelView));
    glUniformMatrix3fv(normalLoc, 1, GL_FALSE, glm::value_ptr(normalMatrix));
    glUniformMatrix4fv(modelViewProjectionLoc, 1, GL_FALSE, glm::value_ptr(modelViewProjection));
}
//----------------------------------------------------------------------------------------------------------------------
void Boat::update(){
    // A random rotation in the z axis to simulate a bobbing effect
    float rndSpeed = 0.5 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.2 - 0.5)));
    if (m_angle >= 0.05){
        m_forwards = false;
    }
    if(m_angle <= -0.05) {
        m_forwards = true;
    }
    float damping = 1.0;
    if(m_forwards){
        if(m_angle > 0.03 || m_angle < -0.03){
            damping = 0.5;
        }
        m_angle += (0.002 * damping) * rndSpeed;
    }
    else{
        if(m_angle < -0.03 || m_angle > 0.03){
            damping = 0.5;
        }
        m_angle -= (0.002*damping) * rndSpeed;
    }

    // A random translation in the y axis to simulate a bobbing effect
    float rndTrans = 0.5 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.2 - 0.5)));
    if (m_yTrans >= 0.05){
        m_upwards = false;
    }
    if(m_yTrans <= -0.05) {
        m_upwards = true;
    }
    damping = 1.0;
    if(m_upwards){
        if(m_yTrans > 0.03 || m_yTrans < -0.03){
            damping = 0.5;
        }
        m_yTrans += (0.002 * damping) * rndTrans;
    }
    else{
        if(m_yTrans < -0.03 || m_yTrans > 0.03){
            damping = 0.5;
        }
        m_yTrans -= (0.002*damping) * rndTrans;
    }

}
//----------------------------------------------------------------------------------------------------------------------
void Boat::render(){
    update();
    m_tex->bind(0);
    m_model->render();
}
//----------------------------------------------------------------------------------------------------------------------
