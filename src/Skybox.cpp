#include "Skybox.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <QGLWidget>
//-------------------------------------------------------------------------------------------------------------------------
Skybox::Skybox(){
    m_sunPos = glm::vec3(0.0, 20.0, -500.0);
    initSkybox();
    createShader();
}
//-------------------------------------------------------------------------------------------------------------------------
Skybox::~Skybox(){
    // Delete skybox model
    delete m_model;
    // Delete skybox shader program
    delete m_shaderProgram;
}
//-------------------------------------------------------------------------------------------------------------------------
void Skybox::initSkybox(){
    m_model = new ModelLoader((char*)"models/cube.obj");
}
//-------------------------------------------------------------------------------------------------------------------------
void Skybox::loadCubeMap(std::string _pathToFile, GLint _activeTexture){
    glActiveTexture(GL_TEXTURE0+_activeTexture);
    GLuint texID;

    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texID);
    const char * suffixes[] = { "posx", "negx", "posy",
       "negy", "posz", "negz" };
    GLuint targets[] = {
       GL_TEXTURE_CUBE_MAP_POSITIVE_X,
       GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
       GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
       GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
       GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
       GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
    };

    for (int i =0; i<6; i++){
       GLuint texture;
       glGenTextures(1, &texture);
       glBindTexture(GL_TEXTURE_2D, texture);
       std::string texName = std::string(_pathToFile) + "_" + suffixes[i] + ".jpg";

       QString name = QString::fromUtf8(texName.c_str());
       QImage *image = new QImage(name);
       QImage tex = QGLWidget::convertToGLFormat(*image);
       glTexImage2D(targets[i],0, GL_RGBA, tex.width(), tex.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tex.bits());
    }

    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}
//-------------------------------------------------------------------------------------------------------------------------
void Skybox::createShader(){
    m_shaderProgram = new ShaderProgram();
    m_vertShader = new Shader("shaders/CubeMapVert.glsl", GL_VERTEX_SHADER);
    m_fragShader =  new Shader("shaders/CubeMapFrag.glsl", GL_FRAGMENT_SHADER);
    m_shaderProgram->attachShader(m_vertShader);
    m_shaderProgram->attachShader(m_fragShader);
    m_shaderProgram->bindFragDataLocation(0, "fragColour");
    m_shaderProgram->link();
    m_shaderProgram->use();

    delete m_vertShader;
    delete m_fragShader;

    loadCubeMap("textures/lake2", 0);

    m_sunPosLoc = m_shaderProgram->getUniformLoc("sunPos");
    GLuint cubeMapLoc = m_shaderProgram->getUniformLoc("cubeMapTex");

    glUniform1i(cubeMapLoc, 0);
    glUniform3f(m_sunPosLoc, m_sunPos.x, m_sunPos.y, m_sunPos.z);
}
//-------------------------------------------------------------------------------------------------------------------------
void Skybox::loadMatricesToShader(glm::mat4 _modelMatrix, glm::mat4 _viewMatrix, glm::mat4 _projectionMatrix){
    m_shaderProgram->use();
    GLuint MVPLoc = m_shaderProgram->getUniformLoc("modelViewProjectionMatrix");

    glm::mat4 modelViewMatrix =_viewMatrix * _modelMatrix;
    glm::mat4 modelViewProjectionMatrix = _projectionMatrix * modelViewMatrix;

    glUniformMatrix4fv(MVPLoc, 1, false, glm::value_ptr(modelViewProjectionMatrix));

}
//----------------------------------------------------------------------------------------------------------------------
void Skybox::update(){
    glUniform3f(m_sunPosLoc, m_sunPos.x, m_sunPos.y, m_sunPos.z);
}
//-------------------------------------------------------------------------------------------------------------------------
void Skybox::render(){
    update();
    m_model->render();
}
//-------------------------------------------------------------------------------------------------------------------------
