#include "OceanGrid.h"
#include "Ocean.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <sys/time.h>
double startTime = 0.0;
//----------------------------------------------------------------------------------------------------------------------
OceanGrid::OceanGrid(int _resolution, int _width, int _depth){
    m_resolution = _resolution;
    m_width = _width;
    m_depth = _depth;
    createShader();
    initialise();
}
//----------------------------------------------------------------------------------------------------------------------
OceanGrid::~OceanGrid(){
    delete m_shaderProgram;
}
//----------------------------------------------------------------------------------------------------------------------
void OceanGrid::createShader(){
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

    glUniform4f(lightPosLoc, 0.0, 5.0, 0.0, 1.0);
    glUniform3f(lightIntLoc, 0.5, 0.5, 0.5);
    glUniform3f(KdLoc, 0.5, 0.5, 0.5);
    glUniform3f(KaLoc, 0.5, 0.5, 0.5);
    glUniform3f(KsLoc, 0.5, 0.5, 0.5);
    glUniform1f(shininessLoc, 100.0);
}
//----------------------------------------------------------------------------------------------------------------------
void OceanGrid::initialise(){
    int width = m_width;
    int depth = m_depth;
    int resolution = m_resolution;

    // calculate the deltas for the x,z values of our point
    float wStep=width/(float)resolution;
    float dStep=depth/(float)resolution;
    // now we assume that the grid is centered at 0,0,0 so we make
    // it flow from -w/2 -d/2
    float xPos=-(width/2.0)+(width/(resolution*2));
    float zPos=-(depth/2.0)+(depth/(resolution*2));
    // now loop from top left to bottom right and generate points

    // Sourced form Jon Macey's NGL library
    for(int z=0; z<=resolution-1; ++z)
    {
      for(int x=0; x<=resolution; ++x)
      {
        // grab the colour and use for the Y (height) only use the red channel
        m_gridVerts.push_back(glm::vec2(xPos,zPos));
        m_gridHeights.push_back(0.0);
        // calculate the new position
        xPos+=wStep;
      }
      // now increment to next z row
      zPos+=dStep;
      // we need to re-set the xpos for new row
      xPos=-(width/2.0)+(width/(resolution*2));
    }

    std::vector <GLuint> indices;
    // some unique index value to indicate we have finished with a row and
    // want to draw a new one
    GLuint restartFlag=resolution*resolution+9999;


    for(int z=0; z<resolution-1; ++z)
    {
      for(int x=0; x<resolution; ++x)
      {
        // Vertex in actual row
        indices.push_back(z  * (resolution+1) + x);
        // Vertex row below
        indices.push_back((z + 1) * (resolution+1) + x);

      }
      // now we have a row of tri strips signal a re-start
      indices.push_back(restartFlag);
    }
    m_vertSize = indices.size();

    // Create our VAO and vertex buffers
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    glGenBuffers(1, &m_VBOverts);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOverts);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2)*m_gridVerts.size(), &m_gridVerts[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    // and one for the index values
    GLuint iboID;
    glGenBuffers(1, &iboID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(GLuint),&indices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &m_VBOheights);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOheights);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*m_gridVerts.size(), &m_gridHeights[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 0, 0);

    registerGLBuffer(m_VBOheights);

    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(restartFlag);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    // Set the start time
    struct timeval tim;
    gettimeofday(&tim, NULL);
    startTime = tim.tv_sec+(tim.tv_usec * 1.0e-6);
}
//----------------------------------------------------------------------------------------------------------------------
void OceanGrid::loadMatricesToShader(glm::mat4 _modelMatrix, glm::mat4 _viewMatrix, glm::mat4 _projectionMatrix){
    m_shaderProgram->use();
    GLuint modelViewLoc = m_shaderProgram->getUniformLoc("modelViewMatrix");
    GLuint normalLoc = m_shaderProgram->getUniformLoc("normalMatrix");
    GLuint projectionLoc = m_shaderProgram->getUniformLoc("projectionMatrix");
    GLuint modelViewProjectionLoc = m_shaderProgram->getUniformLoc("modelViewProjectionMatrix");

    glm::mat4 modelViewMatrix = _viewMatrix * _modelMatrix;
    glm::mat3 normalMatrix = glm::inverseTranspose(glm::mat3(modelViewMatrix));
    glm::mat4 modelViewProjectionMatrix = _projectionMatrix * _viewMatrix * _modelMatrix;

    glUniformMatrix4fv(modelViewLoc, 1, GL_FALSE, glm::value_ptr(modelViewMatrix));
    glUniformMatrix3fv(normalLoc, 1, GL_FALSE, glm::value_ptr(normalMatrix));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(_projectionMatrix));
    glUniformMatrix4fv(modelViewProjectionLoc, 1, GL_FALSE, glm::value_ptr(modelViewProjectionMatrix));
}
void OceanGrid::update(){
    struct timeval tim;
    gettimeofday(&tim, NULL);
    double now = tim.tv_sec+(tim.tv_usec * 1.0e-6);
    updateHeight(now - startTime);
}
//----------------------------------------------------------------------------------------------------------------------
void OceanGrid::render(){
    update();
    glBindVertexArray(m_VAO);
    glPointSize(10.0);
    glDrawElements(GL_POINTS    , m_vertSize,GL_UNSIGNED_INT,0);	// draw first object
    glBindVertexArray(0);
}
//----------------------------------------------------------------------------------------------------------------------
