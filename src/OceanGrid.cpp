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
#include <noise/noise.h>
// ----------------------------------------------------------------------------------------------------------------------------------------
double startTime = 0.0;
// ----------------------------------------------------------------------------------------------------------------------------------------
OceanGrid::OceanGrid(int _resolution, int _width, int _depth){
    m_resolution = _resolution;
    m_width = _width;
    m_depth = _depth;
    m_windSpeed = glm::vec2(1.0, 1.0);
    m_L = 1000;
    m_l = 1.0 / m_L;
    m_A = 0.05;
    m_sunPos = glm::vec3(0.0, 20.0, -500.0);
    m_choppiness = 0.02;
    m_numLayers = 15;
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

    createPerlinTexture(1);

    // Create a texture for storing local reflections
    glGenTextures(1, &m_reflectTex);
    glBindTexture(GL_TEXTURE_2D, m_reflectTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 512, 512, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Vertex Shader uniforms
    GLuint cameraPosLoc = m_shaderProgram->getUniformLoc("cameraPosition");
    glUniform4f(cameraPosLoc, 0.0, 100.0, 100.0, 1.0);

    // Fragment Shader uniforms
    m_sunPositionLoc = m_shaderProgram->getUniformLoc("sun.position");
    GLuint reflectLoc = m_shaderProgram->getUniformLoc("enviromap");
    GLuint sunStrengthLoc = m_shaderProgram->getUniformLoc("sun.strength");
    GLuint sunColourLoc = m_shaderProgram->getUniformLoc("sun.colour");
    GLuint sunShininessLoc = m_shaderProgram->getUniformLoc("sun.shininess");
    GLuint enviromapLoc = m_shaderProgram->getUniformLoc("envirMap");
    GLuint perlinLoc = m_shaderProgram->getUniformLoc("perlinTexture");
    GLuint numLayersLoc = m_shaderProgram->getUniformLoc("numLayers");

    glUniform3f(m_sunPositionLoc, m_sunPos.x, m_sunPos.y, m_sunPos.z);
    glUniform1i(reflectLoc, 0);
    glUniform1f(sunStrengthLoc, 2.0);
    glUniform3f(sunColourLoc, 239.0/255.0, 238.0/255.0, 179.0/255.0);
    glUniform1f(sunShininessLoc, 1.0);
    glUniform1i(enviromapLoc, 0);
    glUniform1i(perlinLoc, 1);
    glUniform1i(numLayersLoc, m_numLayers);

    glBindTexture(GL_TEXTURE_2D, 0);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::createPerlinTexture(int _activeTexture){
    // Create a noise texture used to hide tiling of ocean
    // Noise texture generation taken from chapter 8 of GLSL Cookbook 4.0
    glActiveTexture(GL_TEXTURE0 + _activeTexture);

    int width = 512;
    int height = 512;
    noise::module::Perlin perlinNoise;

    // Base frequency
    perlinNoise.SetFrequency(100.0);
    GLubyte *data = new GLubyte[width * height * 4];
    double xRange = 1.0;
    double yRange = 1.0;
    double xFactor = xRange / width;
    double yFactor = yRange / height;

    for (int oct = 0; oct < 4; oct++){
        perlinNoise.SetOctaveCount(oct+1);
        for (int i=0; i<width; i++){
            for (int j=0; j<height; j++){
                double x = xFactor * i;
                double y = yFactor * j;
                double z = 0.0;
                float val = (float)perlinNoise.GetValue(x, y, z);
                // Scale and translate it between 0 and 1
                val = (val + 1.0f) * 0.5f;
                // Clamp between 0 and 1
                val = val > 1.0f ? 1.0f : val;
                val = val < 0.0f ? 0.0f : val;
                // Store in texture
                data[((j * width + i) * 4) + oct] = (GLubyte) ( val * 255.0f);
            }
        }
    }

    glGenTextures(1, &m_perlinTex);
    glBindTexture(GL_TEXTURE_2D, m_perlinTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindTexture(GL_TEXTURE_2D, 0);

    delete [] data;
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

    glBindTexture(GL_TEXTURE_2D, 0);
}
//----------------------------------------------------------------------------------------------------------------------
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
          xPos+=wStep;
       }

       // now increment to next z row
       zPos+=dStep;
       // we need to re-set the xpos for new row
       xPos=-((float)width/2.0);
    }

    GLuint numTris = (m_resolution-1)*(m_resolution-1)*2;
    GLuint *tris = new GLuint[numTris*3];
    int i, j, fidx = 0;
    for (i=0; i < m_resolution - 1; ++i) {
        for (j=0; j < m_resolution - 1; ++j) {
            tris[fidx*3+0] = i*m_resolution+j; tris[fidx*3+1] = i*m_resolution+j+1; tris[fidx*3+2] = (i+1)*m_resolution+j;
            fidx++;
            tris[fidx*3+0] = i*m_resolution+j+1; tris[fidx*3+1] = (i+1)*m_resolution+j+1; tris[fidx*3+2] = (i+1)*m_resolution+j;
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

    // Indices
    GLuint iboID;
    glGenBuffers(1, &iboID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3*numTris*sizeof(GLuint),tris, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
float OceanGrid::phillips(glm::vec2 _k){
    float kLen = sqrt(_k.x*_k.x + _k.y*_k.y);
    if (kLen== 0.0f){
        return 0.0f;
    }

    float ph = ( (exp( (-1 / ( (kLen * m_L )*(kLen * m_L ) )))) / pow(kLen, 4) );
    ph *= m_A;
    ph *= (glm::normalize(_k).x * m_windSpeed.x + glm::normalize(_k).y * m_windSpeed.y)
            * (glm::normalize(_k).x * m_windSpeed.x + glm::normalize(_k).y * m_windSpeed.y);
    ph *= exp(-kLen * -kLen * m_l * m_l);

    return ph;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
// gaussian random number generator sourced from - NVIDIA OceanFFT
// Generates Gaussian random number with mean 0 and standard deviation 1.
// ----------------------------------------------------------------------------------------------------------------------------------------
float OceanGrid::gauss(){
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
    glm::vec2* h_H0;

    // Assign memory on the host side and device to store h0 and evaluate
    int gridSize = m_resolution*m_resolution;
    h_H0 = (glm::vec2*) malloc(gridSize*sizeof(glm::vec2));
    checkCudaErrors(cudaMalloc((void**)&d_H0, gridSize*sizeof(glm::vec2)));

    for (int m=-m_resolution/2.0; m<m_resolution/2.0; m++){
        for (int n=-m_resolution/2.0; n<m_resolution/2.0; n++){
            glm::vec2 k;
            k.x = (M_2_PI * m) / m_L;
            k.y = (M_2_PI * n) / m_L;
            glm::vec2 h;
            h.x = (1.0/sqrt(2.0)) * gauss() * sqrt(phillips(k));
            h.y = (1.0/sqrt(2.0)) * gauss() * sqrt(phillips(k));
            if (h != h ){
                std::cout<<"hx: "<<h.x<<" hy: "<<h.y<<"m: "<<m<<"n: "<<n<<std::endl;
            }
            h_H0[((n+(m_resolution/2)) + ((m+(m_resolution/2)) * m_resolution))] = h;
        }
    }

    checkCudaErrors(cudaMemcpy(d_H0, h_H0, gridSize*sizeof(glm::vec2), cudaMemcpyHostToDevice));
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::initialise(){
    // Create our VAO and vertex buffers
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    createGrid();

    // Create buffer for tile positions
    std::vector<glm::vec3> tileOffsets;
    for (int x=0; x<m_numLayers; x++){
        for (int z=0; z<m_numLayers; z++){
            tileOffsets.push_back(glm::vec3(x*500 - ((m_numLayers-1)/2)*500.0, 0, z*500- ((m_numLayers-1)/2)*500.0));
        }
    }

    m_numTiles = tileOffsets.size();

    glGenBuffers(1, &m_VBOTilePos);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOTilePos);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*m_numTiles, &tileOffsets[0].x, GL_STATIC_DRAW);
    glEnableVertexAttribArray(3);
    glVertexAttribDivisor(3, 1);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Assign memory for the frequecy field and heights in the time domain
    glm::vec2* h_Ht;
    h_Ht = (glm::vec2*)malloc(m_resolution*m_resolution*sizeof(glm::vec2));
    checkCudaErrors(cudaMalloc((void**)&d_Ht, m_resolution*m_resolution*sizeof(glm::vec2)));

    glm::vec2* h_Heights;
    h_Heights = (glm::vec2*)malloc(m_resolution*m_resolution*sizeof(glm::vec2));
    checkCudaErrors(cudaMalloc((void**)&d_Heights, m_resolution*m_resolution*sizeof(glm::vec2)));

    // Create a texture to store the heights after FFT
    // This is needed because we are then going to combine
    // the FFT height with a perlin noise function to
    // hide tiling
    glGenTextures(1, &m_fftHeightTex);
    glBindTexture(GL_TEXTURE_2D, m_fftHeightTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_resolution, m_resolution, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    checkCudaErrors(cudaGraphicsGLRegisterImage(&m_resourceHeightMap, m_fftHeightTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
    GLuint fftTexLoc = m_shaderProgram->getUniformLoc("fftTexture");

    glUniform1i(fftTexLoc, 2);

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
    glUniform3f(m_sunPositionLoc, m_sunPos.x, m_sunPos.y, m_sunPos.z);

    // The time of the simulation
    struct timeval tim;
    gettimeofday(&tim, NULL);
    double now = tim.tv_sec+(tim.tv_usec * 1.0e-6);

    // Set the time in the shader
    GLuint timeLoc = m_shaderProgram->getUniformLoc("time");
    glUniform1f(timeLoc, now-startTime);

    // Map the graphics resources
    checkCudaErrors(cudaGraphicsMapResources(1, &m_resourceVerts));
    checkCudaErrors(cudaGraphicsMapResources(1, &m_resourceHeightMap));

    // Get pointers to the buffers
    glm::vec3* mapPointerVerts;
    cudaArray* cudaArrayHeightMap;

    size_t numBytes;

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mapPointerVerts, &numBytes, m_resourceVerts));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cudaArrayHeightMap, m_resourceHeightMap, 0, 0));

    cudaResourceDesc viewCudaArrayResourceDesc;
    memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
    viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    viewCudaArrayResourceDesc.res.array.array = cudaArrayHeightMap;

    cudaSurfaceObject_t viewCudaSurfaceObject;
    checkCudaErrors(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));

    // Creates the frequency field
    m_time = now - startTime;
    updateFrequencyDomain(d_H0, (float2*)d_Ht, m_time/2.0, m_resolution);

    // Conduct FFT to retrive heights from frequency domain
    cufftExecC2C(m_fftPlan, (float2*)d_Ht, (float2*)d_Heights, CUFFT_INVERSE);

    // Creates x displacement to the vertex positions
    addChoppiness((float2*)d_Ht, m_resolution);

    // Conduct FFT to retrieve x displacement
    cufftExecC2C(m_fftPlan, (float2*)d_Ht, (float2*)d_Ht, CUFFT_INVERSE);

    // Updates the vertex heights
    updateHeight(mapPointerVerts, viewCudaSurfaceObject, (float2*)d_Heights, (float2*)d_Ht, m_choppiness, m_resolution, 50000);

    // Unmap the cuda graphics resources
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_resourceVerts));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_resourceHeightMap));

    // Syncronise our threads
    cudaThreadSynchronize();

    // Bind our local reflections texture to acitive texture 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_reflectTex);

    // Bind texture for perlin Noise
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_perlinTex);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_fftHeightTex);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::render(){
    update();
    glBindVertexArray(m_VAO);
    glPointSize(10.0);
    glDrawElementsInstanced(GL_TRIANGLES , m_vertSize, GL_UNSIGNED_INT, 0, m_numTiles);

    glBindVertexArray(0);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::moveSunLeft(){
    m_sunPos.x -= 10.0;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::moveSunRight(){
    m_sunPos.x += 10.0;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::moveSunDown(){
    m_sunPos.y -= 10.0;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::moveSunUp(){
    m_sunPos.y += 10.0;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::updateChoppiness(float _choppiness){
    m_choppiness = _choppiness;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void OceanGrid::resetSim(){
    createH0();
}
