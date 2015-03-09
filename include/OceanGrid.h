#ifndef OCEANGRID_H_
#define OCEANGRID_H_

#include <vector>
#include <glm/glm.hpp>

#include <ShaderProgram.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>

class OceanGrid{
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Constructor
    /// @param _resolution the resoluition of the grid
    //----------------------------------------------------------------------------------------------------------------------
    OceanGrid(int _resolution, int _width, int _depth);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Destructor
    //----------------------------------------------------------------------------------------------------------------------
    ~OceanGrid();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Initialise the object and fill m_gridVerts with points;
    //----------------------------------------------------------------------------------------------------------------------
    void initialise();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Load the mode, view projection matrix etc to the shader;
    //----------------------------------------------------------------------------------------------------------------------
    void loadMatricesToShader(glm::mat4 _modelMatrix, glm::mat4 _viewMatrix, glm::mat4 _projectionMatrix);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Render the grid
    //----------------------------------------------------------------------------------------------------------------------
    void render();
    //----------------------------------------------------------------------------------------------------------------------
    void update();
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Create the shader used to draw the grid
    //----------------------------------------------------------------------------------------------------------------------
    void createShader();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Stores the verticies of the grid
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<glm::vec2> m_gridVerts;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Stores the height of the vertices on the grid
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<GLfloat> m_gridHeights;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The resolution of the grid
    //----------------------------------------------------------------------------------------------------------------------
    int m_resolution;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our OpenGL shader program
    //----------------------------------------------------------------------------------------------------------------------
    ShaderProgram *m_shaderProgram;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Vertex Shader
    //----------------------------------------------------------------------------------------------------------------------
    Shader *m_vertShader;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Fragment Shader
    //----------------------------------------------------------------------------------------------------------------------
    Shader *m_fragShader;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our OpenGL Vertex Array Object
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_VAO;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief A buffer to store the vertex positions of our grid
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_VBOverts;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The number of points on the grid
    //----------------------------------------------------------------------------------------------------------------------
    int m_vertSize;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The width of the grid
    //----------------------------------------------------------------------------------------------------------------------
    int m_width;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The depth of the grid
    //----------------------------------------------------------------------------------------------------------------------
    int m_depth;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief A buffer to store the heights of the points on the grid
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_VBOheights;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief A buffer to store the colours of the points on the grid
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_VBOcolours;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief A buffer to store the normals of the points on the grid
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_VBOnormals;
    //----------------------------------------------------------------------------------------------------------------------
//    cudaGraphicsResource_t m_res;
    cudaGraphicsResource_t m_resourceHeight;
    cudaGraphicsResource_t m_resourceNormal;

};


#endif
