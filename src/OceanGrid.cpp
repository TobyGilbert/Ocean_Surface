// ----------------------------------------------------------------------------------------------------------------------------------------
/// @author Toby Gilbert
// ----------------------------------------------------------------------------------------------------------------------------------------
#include "OceanGrid.h"
#include "Ocean.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <sys/time.h>
#include <complex>
#include <QImage>
#include <QGLWidget>
// ----------------------------------------------------------------------------------------------------------------------------------------
double startTime = 0.0;
// ----------------------------------------------------------------------------------------------------------------------------------------
OceanGrid::OceanGrid(int _resolution, int _width, int _depth){
    m_resolution = _resolution;
    m_width = _width;
    m_depth = _depth;
    m_windSpeed = glm::vec2(5.0, 5.0);
    m_L = 1000;
    m_l = 1 / m_L;
    m_A = 0.1;
    m_sunPos = glm::vec3(0.0, -0.5, 0.5);
    m_choppiness = 0.02;
    createShader();
    initialise();
}
// ----------------------------------------------------------------------------------------------------------------------------------------
OceanGrid::~OceanGrid(){
    delete m_shaderProgram;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::createShader(){
    m_shaderProgram = new ShaderProgram();
    m_vertShader = new Shader("shaders/OceanVert.glsl", GL_VERTEX_SHADER);
    m_fragShader = new Shader("shaders/OceanFrag.glsl", GL_FRAGMENT_SHADER);
    m_shaderProgram->attachShader(m_vertShader);
    m_shaderProgram->attachShader(m_fragShader);
    m_shaderProgram->bindFragDataLocation(0, "fragColour");
    m_shaderProgram->link();
    m_shaderProgram->use();

    delete m_vertShader;
    delete m_fragShader;

    loadCubeMap("textures/miramar", 0);

    // Vertex Shader uniforms
    GLuint cameraPosLoc = m_shaderProgram->getUniformLoc("cameraPosition");
    glUniform4f(cameraPosLoc, 0.0, 50.0, 50.0, 1.0);

    // Fragment Shader uniforms
    GLuint sunStrengthLoc = m_shaderProgram->getUniformLoc("sun.strength");
    GLuint sunColourLoc = m_shaderProgram->getUniformLoc("sun.colour");
    m_sunDirectionLoc = m_shaderProgram->getUniformLoc("sun.direction");
    GLuint sunShininessLoc = m_shaderProgram->getUniformLoc("sun.shininess");
    GLuint enviromapLoc = m_shaderProgram->getUniformLoc("envirMap");

    glUniform1f(sunStrengthLoc, 0.3);
    glUniform3f(sunColourLoc, 255.0/255.0, 255.0/255.0, 255.0/255.0);
    glUniform3f(m_sunDirectionLoc, m_sunPos.x, m_sunPos.y, m_sunPos.z);
    glUniform1f(sunShininessLoc, 0.3);
    glUniform1i(enviromapLoc, 0);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::loadCubeMap(std::string _pathToFile, GLint _activeTexture){
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

    for (int i=0; i<6; i++){
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

// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::initialise(){
    int width = m_width;
    int depth = m_depth;
    int resolution = m_resolution;
    std::vector<glm::vec3> normals;

    // calculate the deltas for the x,z values of our point
    float wStep=(float)width/(float)resolution;
    float dStep=(float)depth/(float)resolution;
    // now we assume that the grid is centered at 0,0,0 so we make
    // it flow from -w/2 -d/2
    float xPos=-((float)width/2.0);
    float zPos=-((float)depth/2.0);
    // now loop from top left to bottom right and generate points

    // Sourced form Jon Macey's NGL library
    for(int z=0; z<resolution; z++)
    {
      for(int x=0; x<resolution; x++)
      {
        // grab the colour and use for the Y (height) only use the red channel
        m_gridVerts.push_back(glm::vec3(xPos,0.0, zPos));
        m_gridHeights.push_back(0.0);
        normals.push_back(glm::vec3(0.0, 0.0, 0.0));
        // calculate the new position
        xPos+=wStep;
      }
      // now increment to next z row
      zPos+=dStep;
      // we need to re-set the xpos for new row
      xPos=-((float)width/2.0);
    }

    int res = m_resolution;
    unsigned int num_tris = (res-1)*(res-1)*2;
    GLuint *tris = new GLuint[num_tris*3];
    int i, j, fidx = 0;
    for (i=0; i < res - 1; ++i) {
        for (j=0; j < res - 1; ++j) {
            tris[fidx*3+0] = i*res+j; tris[fidx*3+1] = i*res+j+1; tris[fidx*3+2] = (i+1)*res+j;
            fidx++;
            tris[fidx*3+0] = i*res+j+1; tris[fidx*3+1] = (i+1)*res+j+1; tris[fidx*3+2] = (i+1)*res+j;
            fidx++;
        }
    }

    m_vertSize = 3*num_tris;

    // Create our VAO and vertex buffers
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    // Vertices
    glGenBuffers(1, &m_VBOverts);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOverts);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*m_gridVerts.size(), &m_gridVerts[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceVerts, m_VBOverts, cudaGraphicsRegisterFlagsNone));

    // Normals
    glGenBuffers(1, &m_VBOnormals);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOnormals);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*normals.size(), &normals[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceNormals, m_VBOnormals, cudaGraphicsRegisterFlagsNone));

    // Indices
    GLuint iboID;
    glGenBuffers(1, &iboID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3*num_tris*sizeof(GLuint),tris, GL_STATIC_DRAW);

    // Frequency field
    glGenBuffers(1, &m_VBOHt);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOHt);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2)*m_gridVerts.size(), &m_gridHeights[0], GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceHt, m_VBOHt, cudaGraphicsRegisterFlagsNone));

    // Heights after FFT
    glGenBuffers(1, &m_VBOheights);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOheights);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2)*m_gridVerts.size(), 0, GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceHeights, m_VBOheights, cudaGraphicsRegisterFlagsNone));

    // Create a set of time dependent amplitude and phases
    createH0();

    std::vector<wave> waves;
    wave w1;
    w1.D = glm::vec2(1.0, 0.0);
    w1.W = sqrt(double(9.81 * w1.D.length()));//0.5*M_PI;
    w1.A = 0.25f*1.0f/8.0f;
    w1.Q = float(1.0/(w1.W * w1.A));
    w1.S = 0.1;
    w1.L = 1.0;
    w1.phaseConstant = 0.0;

    waves.push_back(w1);

    wave w2;
    w2.D = glm::vec2(0.0, 1.0);
    w2.W =  sqrt(double(9.81 * w2.D.length()));//0.2f*M_PI;
    w2.A = 0.25f*1.0f/8.0f;
    w2.Q = float((1.0/(w2.W * w2.A)));
    w2.S = 0.2;
    w2.L = 1.0;
    w2.phaseConstant =0.0;

    waves.push_back(w2);

    wave w3;
    w3.D = glm::vec2(-1.0, 1.0);
    w3.W = sqrt(double(9.81 * w3.D.length()));//2.0f*M_PI;
    w3.A = 1.0f/100.0f;
    w3.Q = float((1.0/(w3.W * w3.A)));
    w3.S = 0.3;
    w3.L = 0.5;
    w3.phaseConstant = 1.0;

    waves.push_back(w3);

    glGenBuffers(1, &m_VBOwaves);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOwaves);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*8*waves.size(), &waves[0], GL_STATIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceWaves, m_VBOwaves, cudaGraphicsRegisterFlagsReadOnly));

    // Create reflection texture
    glGenTextures(1, &m_reflectTex);
    glBindTexture(GL_TEXTURE_2D, m_reflectTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 512, 512, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glUniform1i(m_reflectLoc, 0);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // create FFT plan
    if(cufftPlan2d(&m_fftPlan, 256, 256, CUFFT_C2C) != CUFFT_SUCCESS) { printf("Cuda: cufftPlan1d CUFFT_C2C failed\n"); }

    // Set the start time
    struct timeval tim;
    gettimeofday(&tim, NULL);
    startTime = tim.tv_sec+(tim.tv_usec * 1.0e-6);
}

// ----------------------------------------------------------------------------------------------------------------------------------------
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
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::update(){
    // Set the direction of the sun
    glUniform3f(m_sunDirectionLoc, m_sunPos.x, m_sunPos.y, m_sunPos.z);

    // The time of the simulation
    struct timeval tim;
    gettimeofday(&tim, NULL);
    double now = tim.tv_sec+(tim.tv_usec * 1.0e-6);

    // Map the graphics resources
    cudaGraphicsMapResources(1, &m_resourceVerts);
    cudaGraphicsMapResources(1, &m_resourceNormals);
    cudaGraphicsMapResources(1, &m_resourceWaves);
    cudaGraphicsMapResources(1, &m_resourceHt);
    cudaGraphicsMapResources(1, &m_resourceH0);
    cudaGraphicsMapResources(1, &m_resourceHeights);

    // Get pointers to the buffers
    glm::vec3* mapPointerVerts;
    glm::vec3* mapPointerNormals;
    wave* mapPointerWaves;
    glm::vec2* mapPointerH0;
    float2* mapPointerHt;
    float2* mapPointerHeights;

    size_t numBytes;

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mapPointerVerts, &numBytes, m_resourceVerts));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mapPointerNormals, &numBytes, m_resourceNormals));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mapPointerWaves, &numBytes, m_resourceWaves));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mapPointerHt, &numBytes, m_resourceHt));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mapPointerH0, &numBytes, m_resourceH0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mapPointerHeights, &numBytes, m_resourceHeights));

    // Creates the frequency field

    m_time = now - startTime;
    updateFrequencyDomain(mapPointerH0, mapPointerHt, m_time, m_resolution);
    // Conduct FFT to retrive heights from frequency domain
    cufftExecC2C(m_fftPlan, mapPointerHt, mapPointerHeights, CUFFT_INVERSE);

    // Creates x displacement to the vertex positions
    addChoppiness(mapPointerHt, m_resolution);

    // Conduct FFT to retrieve x displacement
    cufftExecC2C(m_fftPlan, mapPointerHt, mapPointerHt, CUFFT_INVERSE);

    // Updates the vertex heights
    updateHeight(mapPointerVerts, mapPointerHeights, mapPointerNormals, mapPointerHt, m_choppiness, m_resolution, 50000);

    // Gerstner
//    updateGerstner(mapPointerVerts, mapPointerNormals, mapPointerWaves, (now -startTime), m_resolution, 3);

    // Unmap the cuda graphics resources
    cudaGraphicsUnmapResources(1, &m_resourceVerts);
    cudaGraphicsUnmapResources(1, &m_resourceNormals);
    cudaGraphicsUnmapResources(1, &m_resourceWaves);
    cudaGraphicsUnmapResources(1, &m_resourceHt);
    cudaGraphicsUnmapResources(1, &m_resourceH0);
    cudaGraphicsUnmapResources(1, &m_resourceHeights);

    // Syncronise our threads
    cudaThreadSynchronize();

    // Bind our local reflections texture to acitive texture 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_reflectTex);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
float OceanGrid::phillips(glm::vec2 _k){
    float kLen = sqrt(_k.x*_k.x + _k.y*_k.y);
    if (kLen== 0.0f)
    {
        return 0.0f;
    }
    float ph = ( (exp( (-1 / ( (kLen * m_L )*(kLen * m_L ) )))) / pow(kLen, 4) );
    ph *= m_A;
    ph *= (glm::normalize(_k).x * glm::normalize(m_windSpeed).x + glm::normalize(_k).y * glm::normalize(m_windSpeed).y)
            * (glm::normalize(_k).x * glm::normalize(m_windSpeed).x + glm::normalize(_k).y * glm::normalize(m_windSpeed).y);
    ph *= exp(-kLen * -kLen * m_l * m_l);
    return ph;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
// gaussian random number generator sourced from - NVIDIA OceanFFT
// Generates Gaussian random number with mean 0 and standard deviation 1.
// ----------------------------------------------------------------------------------------------------------------------------------------
float OceanGrid::gauss()
{
    float u1 = rand() / (float)RAND_MAX;
    float u2 = rand() / (float)RAND_MAX;

    if (u1 < 1e-6f)
    {
        u1 = 1e-6f;
    }

    return sqrtf(-2 * logf(u1)) * cosf(2*M_PI * u2);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::createH0(){
    std::vector<glm::vec2> h0;
    for (int m=-m_resolution/2; m<m_resolution/2; m++){
        for (int n=-m_resolution/2; n<m_resolution/2; n++){
            glm::vec2 k;
            k.x = (M_2_PI * m) / m_L;
            k.y = (M_2_PI * n) / m_L;
            glm::vec2 h;
            h.x = (1.0/sqrt(2.0)) * gauss() * sqrt(phillips(k));
            h.y = (1.0/sqrt(2.0)) * gauss() * sqrt(phillips(k));
            if (h != h ){
                std::cout<<"hx: "<<h.x<<" hy: "<<h.y<<"m: "<<m<<"n: "<<n<<std::endl;
            }
            h0.push_back(h);
        }
    }
    glGenBuffers(1, &m_VBOh0);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOh0);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2)*h0.size(), &h0[0], GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceH0, m_VBOh0, cudaGraphicsRegisterFlagsNone));
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::render(){
    update();
    glBindVertexArray(m_VAO);
    glPointSize(10.0);
    glDrawElements(GL_TRIANGLES , m_vertSize, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::moveSunLeft(){
    m_sunPos.x -= 0.1;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::moveSunRight(){
    m_sunPos.x += 0.1;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::moveSunDown(){
    m_sunPos.y -= 0.1;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::moveSunUp(){
    m_sunPos.y += 0.1;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::updateChoppiness(float _choppiness){
    m_choppiness = _choppiness;
}
