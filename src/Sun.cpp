#include "Sun.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>

Sun::Sun(){
    createShader();
    initialise();
}

Sun::~Sun(){
    delete m_shaderProgram;
}

void Sun::createShader(){
    m_shaderProgram = new ShaderProgram();
    m_vertShader = new Shader("shaders/PhongVert.glsl", GL_VERTEX_SHADER);
    m_fragShader = new Shader("shaders/PhongFrag.glsl", GL_FRAGMENT_SHADER);
    m_shaderProgram->attachShader(m_vertShader);
    m_shaderProgram->attachShader(m_fragShader);
    m_shaderProgram->bindFragDataLocation(0, "fragColour");
    m_shaderProgram->link();
    m_shaderProgram->use();

    delete m_vertShader;
    delete m_fragShader;

    GLuint lightPosLoc = m_shaderProgram->getUniformLoc("light.position");
    GLuint lightIntLoc = m_shaderProgram->getUniformLoc("light.intensity");
    GLuint KdLoc = m_shaderProgram->getUniformLoc("Kd");
    GLuint KaLoc = m_shaderProgram->getUniformLoc("Ka");
    GLuint KsLoc = m_shaderProgram->getUniformLoc("Ks");
    GLuint shininessLoc = m_shaderProgram->getUniformLoc("shininess");

    glUniform4f(lightPosLoc, 0.0, 10.0, 50.0, 1.0);
    glUniform3f(lightIntLoc, 1.0, 1.0, 1.0);
    glUniform3f(KdLoc, 0.3, 0.3, 0.3);
    glUniform3f(KaLoc, 0.3, 0.3, 0.3);
    glUniform3f(KsLoc, 0.3, 0.3, 0.3);
    glUniform1f(shininessLoc, 100.0);
}

void Sun::initialise(){
    m_model = new ModelLoader("models/sphere.obj");
}

void Sun::loadMatricesToShader(glm::mat4 _modelMatrix, glm::mat4 _viewMatrix, glm::mat4 _projectionMatrix){
    m_shaderProgram->use();
    GLuint modelViewLoc = m_shaderProgram->getUniformLoc("modelViewMatrix");
    GLuint modelViewProjectionLoc = m_shaderProgram->getUniformLoc("modelViewProjectionMatrix");
    GLuint normalMatrixLoc = m_shaderProgram->getUniformLoc("normalMatrix");

    glm::mat4 modelView = _viewMatrix * _modelMatrix;
    glm::mat4 modelViewProjection = _projectionMatrix * modelView;
    glm::mat3 normal = glm::inverseTranspose(glm::mat3(modelView));

    glUniformMatrix4fv(modelViewLoc, 1, GL_FALSE, glm::value_ptr(modelView));
    glUniformMatrix4fv(modelViewProjectionLoc, 1, GL_FALSE, glm::value_ptr(modelViewProjection));
    glUniformMatrix3fv(normalMatrixLoc, 1, GL_FALSE, glm::value_ptr(normal));
}

void Sun::render(){
    m_model->render();
}
