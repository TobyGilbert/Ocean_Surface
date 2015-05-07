//-------------------------------------------------------------------------------------------------------------------------
/// @author Toby Gilbert
//-------------------------------------------------------------------------------------------------------------------------
#include "OceanGrid.h"
#include "Ocean.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <sys/time.h>
#include <complex>
#include <QImage>
#include <QGLWidget>
#include <helper_math.h>
#include "MathsUtils.h"
#ifdef DARWIN
#include <noise/noise.h>
#else
#include <libnoise/noise.h>
#endif
//-------------------------------------------------------------------------------------------------------------------------
double startTime = 0.0;
//-------------------------------------------------------------------------------------------------------------------------
OceanGrid::OceanGrid(int _resolution, int _width, int _depth){
    m_resolution = _resolution;
    m_width = _width;
    m_depth = _depth;
    m_windSpeed = 100.0;
    m_windDirection = make_float2(0.0, 0.5);
    m_L = (m_windSpeed*m_windSpeed) / 9.81;
    m_l = 1.0 / m_L;
    m_A = 5.0;
    m_sunPos = glm::vec3(0.0, 20.0, -500.0);
    m_choppiness = 0.02;
    m_numLayers = 10;
    m_seaBaseColour = make_float3(0.1,0.19,0.22);
    m_seaTopColour = make_float3(0.8,0.9,0.6);
    m_sunStreak = 200.0;
//    m_seaBaseColour = make_float3( 54.0/255.0, 60.0/255.0, 74.0/255.0);
//    m_seaTopColour = make_float3(  127.0/255.0, 160.0/255.0, 205.0/255.0);
    createShader();
    initialise();
}
//-------------------------------------------------------------------------------------------------------------------------
OceanGrid::~OceanGrid(){
    delete m_shaderProgram;
    delete m_dudvTex;
    cudaFree(d_Ht);
    cudaFree(d_H0);
    cudaFree(d_Heights);
    cudaFree(d_chopX);
    cudaFree(d_chopZ);
    free(h_Ht);
    free(h_H0);
}
//-------------------------------------------------------------------------------------------------------------------------
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

    // Create a texture for storing local reflections
    glGenTextures(1, &m_reflectTex);
    glBindTexture(GL_TEXTURE_2D, m_reflectTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1024, 1024, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // dudv map
    m_dudvTex = new Texture("textures/water_dudv.jpg");
    m_dudvTex->bind(1);

    // Vertex Shader uniforms
    GLuint cameraPosLoc = m_shaderProgram->getUniformLoc("cameraPosition");
    glUniform4f(cameraPosLoc, 0.0, 100.0, 100.0, 1.0);

    // Fragment Shader uniforms
    m_timeLoc = m_shaderProgram->getUniformLoc("time");
    m_seaBaseColLoc = m_shaderProgram->getUniformLoc("seaBaseColour");
    m_seaTopColLoc = m_shaderProgram->getUniformLoc("seaTopColour");
    m_sunPositionLoc = m_shaderProgram->getUniformLoc("sun.position");
    m_sunStreakLoc = m_shaderProgram->getUniformLoc("sun.streak");
    GLuint reflectLoc = m_shaderProgram->getUniformLoc("reflectTex");
    GLuint enviromapLoc = m_shaderProgram->getUniformLoc("envirMap");
    GLuint numLayersLoc = m_shaderProgram->getUniformLoc("numLayers");
    GLuint dudvMapLoc = m_shaderProgram->getUniformLoc("dudvTex");
    GLuint fogFarLoc = m_shaderProgram->getUniformLoc("fogFarDist");
    GLuint fogNearLoc = m_shaderProgram->getUniformLoc("fogNearDist");

    glUniform1f(m_timeLoc, 0.0);
    glUniform3f(m_seaBaseColLoc, m_seaBaseColour.x, m_seaBaseColour.y, m_seaBaseColour.z);
    glUniform3f(m_seaTopColLoc,m_seaTopColour.x, m_seaTopColour.y, m_seaTopColour.z);
    glUniform3f(m_sunPositionLoc, m_sunPos.x, m_sunPos.y, m_sunPos.z);
    glUniform1f(m_sunStreakLoc, m_sunStreak);
    glUniform1i(reflectLoc, 0);
    glUniform1i(enviromapLoc, 0);
    glUniform1i(numLayersLoc, m_numLayers);
    glUniform1i(dudvMapLoc, 1);
    glUniform1f(fogNearLoc, 750.0);
    glUniform1f(fogFarLoc, 3000.0);

    glBindTexture(GL_TEXTURE_2D, 0);
}
//-------------------------------------------------------------------------------------------------------------------------
void OceanGrid::createGrid(){
    std::vector<glm::vec3> vertices, normals;

    int width = m_width;
    int depth = m_depth;

    // calculate the deltas for the x,z values of our point
    float wStep=(float)width/(float)m_resolution;
    float dStep=(float)depth/(float)m_resolution;
    // now we assume that the grid is centered at 0,0,0 so we make
    // it flow from -w/2 -d/2
    float xPos=-((float)width/2.0);
    float zPos=-((float)depth/2.0);
    // now loop from top left to bottom right and generate points

    // Sourced form Jon Macey's NGL library
    for(int z=0; z<m_resolution; z++){
       for(int x=0; x<m_resolution; x++){

           // grab the colour and use for the Y (height) only use the red channel
          vertices.push_back(glm::vec3(xPos,0.0, zPos));
          normals.push_back(glm::vec3(0.0, 0.0, 0.0));

          // calculate the new position
          zPos+=dStep;
       }

       // now increment to next z row
       xPos+=wStep;
       // we need to re-set the xpos for new row
       zPos=-((float)depth/2.0);
    }

    GLuint numTris = (m_resolution-1)*(m_resolution-1)*2;
    GLuint *tris = new GLuint[numTris*3];
    int i, j, fidx = 0;
    for (i=0; i < m_resolution - 1; ++i) {
        for (j=0; j < m_resolution - 1; ++j) {
            tris[fidx*3+0] = (i+1)*m_resolution+j;
            tris[fidx*3+1] = i*m_resolution+j+1;
            tris[fidx*3+2] = i*m_resolution+j;
            fidx++;
            tris[fidx*3+0] = (i+1)*m_resolution+j;
            tris[fidx*3+1] = (i+1)*m_resolution+j+1;
            tris[fidx*3+2] = i*m_resolution+j+1;
            fidx++;
        }
    }

    m_vertSize = 3*numTris;

    // Vertices
    glGenBuffers(1, &m_VBOverts);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOverts);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*vertices.size(), &vertices[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceVerts, m_VBOverts, cudaGraphicsRegisterFlagsNone));

    std::vector<float3> norms;
    for (int i=0; i<m_resolution*m_resolution; i++){
        norms.push_back(make_float3(0.0, 1.0, 0.0));
    }

    // Normals
    glGenBuffers(1, &m_VBOnorms);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOnorms);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*norms.size(), &norms[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceNorms, m_VBOnorms, cudaGraphicsRegisterFlagsWriteDiscard));

    // Indices
    GLuint iboID;
    glGenBuffers(1, &iboID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3*numTris*sizeof(GLuint),tris, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
//-------------------------------------------------------------------------------------------------------------------------
float OceanGrid::phillips(float2 _k){
    float kLen = sqrt(_k.x*_k.x + _k.y*_k.y);
    if (kLen == 0.0f){
        return 0.0f;
    }
    float ph = ( exp( -1 / ( (kLen * m_L )*(kLen * m_L ) ))  / pow(kLen, 4) );
    ph *= m_A;


    // Waves moving in the opposite direction to the wind get dampened
    if (((_k.x * m_windDirection.x) + (_k.y * m_windDirection.y))  < 0.0){
        ph *= 0.05;
    }

    // | k . w |^2
    float kw =  dot(normalise(_k), normalise(m_windDirection));
    ph *= kw * kw;

    return ph;
}
//-------------------------------------------------------------------------------------------------------------------------
// gaussian random number generator sourced from - NVIDIA OceanFFT
// Generates Gaussian random number with mean 0 and standard deviation 1.
//-------------------------------------------------------------------------------------------------------------------------
float OceanGrid::gauss(){
    float u1 = rand() / (float)RAND_MAX;
    float u2 = rand() / (float)RAND_MAX;

    if (u1 < 1e-6f)
    {
        u1 = 1e-6f;
    }

    return sqrtf(-2 * logf(u1)) * cosf(2*M_PI * u2);
}
//-------------------------------------------------------------------------------------------------------------------------
void OceanGrid::createH0(){
    // Assign memory on the host side and device to store h0 and evaluate
    int gridSize = m_resolution*m_resolution;

    for (int m=-m_resolution/2.0; m<m_resolution/2.0; m++){
        for (int n=-m_resolution/2.0; n<m_resolution/2.0; n++){
            float2 k;
            k.x = (M_2_PI * m) / 1000.0;
            k.y = (M_2_PI * n) / 1000.0;
            float2 h;
            h.x = (1.0/sqrt(2.0)) * gauss() * sqrt(phillips(k));
            h.y = (1.0/sqrt(2.0)) * gauss() * sqrt(phillips(k));
            h_H0[((n+(m_resolution/2)) + ((m+(m_resolution/2)) * m_resolution))] = h;
        }
    }
    checkCudaErrors(cudaMemcpy(d_H0, h_H0, gridSize*sizeof(glm::vec2), cudaMemcpyHostToDevice));
}
//-------------------------------------------------------------------------------------------------------------------------
void OceanGrid::initialise(){
    // Create our VAO and vertex buffers
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    createGrid();

    int offset = 495;

    // Create buffer for tile positions
    std::vector<glm::vec3> tileOffsets;
    for (int x=0; x<m_numLayers; x++){
        for (int z=0; z<m_numLayers; z++){
            tileOffsets.push_back(glm::vec3(x*offset - ((m_numLayers-1)/2)*offset, 0, z*offset- ((m_numLayers-1)/2)*offset));
        }
    }

    m_numTiles = tileOffsets.size();

    glGenBuffers(1, &m_VBOTilePos);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOTilePos);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*m_numTiles, &tileOffsets[0].x, GL_STATIC_DRAW);
    glEnableVertexAttribArray(3);
#ifdef DARWIN
    glVertexAttribDivisor(3, 1);
#else
    glVertexAttribDivisorARB(3, 1);
#endif
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Assign memory for the frequecy field and heights in the time domain
    h_Ht = (float2*)malloc(m_resolution*m_resolution*sizeof(float2));
    checkCudaErrors(cudaMalloc((void**)&d_Ht, m_resolution*m_resolution*sizeof(float2)));

    checkCudaErrors(cudaMalloc((void**)&d_Heights, m_resolution*m_resolution*sizeof(float2)));

    int gridSize = m_resolution*m_resolution;
    h_H0 = (float2*) malloc(gridSize*sizeof(float2));
    checkCudaErrors(cudaMalloc((void**)&d_H0, gridSize*sizeof(float2)));

    checkCudaErrors(cudaMalloc((void**)&d_chopX, gridSize*sizeof(float2)));
    checkCudaErrors(cudaMalloc((void**)&d_chopZ, gridSize*sizeof(float2)));

    // Create a set of time dependent amplitude and phases
    createH0();

    // create FFT plan
    if(cufftPlan2d(&m_fftPlan, m_resolution, m_resolution, CUFFT_C2C) != CUFFT_SUCCESS) {
        printf("Cuda: cufftPlan2d CUFFT_C2C failed\n");
    }

    // Set the start time
    struct timeval tim;
    gettimeofday(&tim, NULL);
    startTime = tim.tv_sec+(tim.tv_usec * 1.0e-6);
}
//-------------------------------------------------------------------------------------------------------------------------
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
//-------------------------------------------------------------------------------------------------------------------------
void OceanGrid::update(){
    // Set the sea colours in the shader which can be set in the GUI
    glUniform3f(m_seaBaseColLoc, m_seaBaseColour.x, m_seaBaseColour.y, m_seaBaseColour.z);
    glUniform3f(m_seaTopColLoc, m_seaTopColour.x, m_seaTopColour.y, m_seaTopColour.z);

    // Set the direction of the sun
    glUniform3f(m_sunPositionLoc, m_sunPos.x, m_sunPos.y, m_sunPos.z);

    // Sets the sun streak width
    glUniform1f(m_sunStreakLoc, m_sunStreak);

    // The time of the simulation
    struct timeval tim;
    gettimeofday(&tim, NULL);
    double now = tim.tv_sec+(tim.tv_usec * 1.0e-6);
    glUniform1f(m_timeLoc, now);

    // Set the time in the shader
    GLuint timeLoc = m_shaderProgram->getUniformLoc("time");
    glUniform1f(timeLoc, now-startTime);

    // Map the graphics resources
    checkCudaErrors(cudaGraphicsMapResources(1, &m_resourceVerts));
    checkCudaErrors(cudaGraphicsMapResources(1, &m_resourceNorms));

    // Get pointers to the buffers
    float3* mapPointerVerts;
    size_t numBytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mapPointerVerts, &numBytes, m_resourceVerts));

    float3* mapPointerNorms;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mapPointerNorms, &numBytes, m_resourceNorms));

    // Creates the frequency field
    m_time = now - startTime;
    updateFrequencyDomain(d_H0, d_Ht, m_time/2.0, m_resolution);

    // Syncronise our threads
    cudaThreadSynchronize();

    // Conduct FFT to retrive heights from frequency domain
    cufftExecC2C(m_fftPlan, d_Ht, d_Heights, CUFFT_INVERSE);

    // Syncronise our threads
    cudaThreadSynchronize();

    // Creates x displacement to the vertex positions
    addChoppiness(d_Ht, d_chopX, d_chopZ,  m_resolution, make_float2(m_windDirection.x, m_windDirection.y));
    cufftExecC2C(m_fftPlan, d_chopX, d_chopX, CUFFT_INVERSE);
    cufftExecC2C(m_fftPlan, d_chopZ, d_chopZ, CUFFT_INVERSE);

    // Updates the vertex heights
    updateHeight(mapPointerVerts, mapPointerNorms, d_Heights, d_chopX, d_chopZ, m_choppiness, m_resolution, (500000.0));

    // Unmap the cuda graphics resources
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_resourceVerts));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_resourceNorms));

    // Syncronise our threads
    cudaThreadSynchronize();

    // Bind our local reflections texture to acitive texture 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_reflectTex);
    glActiveTexture(GL_TEXTURE1);
    m_dudvTex->bind(1);
}
//-------------------------------------------------------------------------------------------------------------------------
void OceanGrid::render(){
    update();
    glBindVertexArray(m_VAO);
    glPointSize(3.0);
    glDrawElementsInstanced(GL_TRIANGLES , m_vertSize, GL_UNSIGNED_INT, 0, m_numTiles);

    glBindVertexArray(0);
}
//-------------------------------------------------------------------------------------------------------------------------
void OceanGrid::updateChoppiness(float _choppiness){
    m_choppiness = _choppiness;
}
//-------------------------------------------------------------------------------------------------------------------------
void OceanGrid::resetSim(){
    free(h_H0);
    cudaFree(d_H0);
    free(h_Ht);
    cudaFree(d_Ht);
    cudaFree(d_Heights);
    cudaFree(d_chopX);
    cudaFree(d_chopZ);

    glDeleteBuffers(1, &m_VBOverts);
    glDeleteBuffers(1, &m_VBOnorms);

    initialise();
}
//-------------------------------------------------------------------------------------------------------------------------
void OceanGrid::setResolution(int _resolution){
    // Set out new resolution
    m_resolution = _resolution;

    // Free our memory so we can reinstantiate it at a new size
    resetSim();
}
//-------------------------------------------------------------------------------------------------------------------------
void OceanGrid::setAmplitude(double _value){
    // Set out new amplitude
    m_A = _value;

    // Free our memory so we can reinstantiate it with a new amplitude
    resetSim();
}
//-------------------------------------------------------------------------------------------------------------------------
void OceanGrid::setSeaBaseCol(QColor _col){
    m_seaBaseColour = make_float3(_col.red()/255.0, _col.green()/255.0, _col.blue()/255.0);
}
//-------------------------------------------------------------------------------------------------------------------------
void OceanGrid::setSeaTopCol(QColor _col){
    m_seaTopColour = make_float3(_col.red()/255.0, _col.green()/255.0, _col.blue()/255.0);
}
//-------------------------------------------------------------------------------------------------------------------------
void OceanGrid::setSunStreakWidth(float _width){
    m_sunStreak = _width;
}
//-------------------------------------------------------------------------------------------------------------------------
