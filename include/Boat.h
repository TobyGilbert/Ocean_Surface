/** @addtogroup OceanFFTStandAlone */
/*@{*/
//----------------------------------------------------------------------------------------------------------------------
#ifndef BOAT_H
#define BOAT_H
//----------------------------------------------------------------------------------------------------------------------
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
//----------------------------------------------------------------------------------------------------------------------
#include "ShaderProgram.h"
#include "ModelLoader.h"
#include "Texture.h"
//----------------------------------------------------------------------------------------------------------------------
class Boat{
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Constructor
    //----------------------------------------------------------------------------------------------------------------------
    Boat(std::string _model, std::string _texture);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Destructor
    //----------------------------------------------------------------------------------------------------------------------
    ~Boat();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Initialises the boat obj model
    //----------------------------------------------------------------------------------------------------------------------
    void initialise(std::string _model);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Load the matrices to the shader for drawing
    /// @param _modelMatrix the model matrix
    /// @param _viewMatrix the view matrix
    /// @param _projectionMatrix the projection matrix
    //----------------------------------------------------------------------------------------------------------------------
    void loadMatricesToShader(glm::mat4 _modelMatrix, glm::mat4 _viewMatrix, glm::mat4 _projectionMatrix);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Load the matrices to the shader for drawing a the boat using a clipping plane
    /// @param _modelMatrix the model matrix
    /// @param _viewMatrix the view matrix
    /// @param _projectionMatrix the projection matrix
    //----------------------------------------------------------------------------------------------------------------------
    void loadMatricesToShaderClipped(glm::mat4 _modelMatrix, glm::mat4 _viewMatrix, glm::mat4 _projectionMatrix);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Renders the boat obj model
    //----------------------------------------------------------------------------------------------------------------------
    void render();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Update the boat bobbing action
    //----------------------------------------------------------------------------------------------------------------------
    void update();
    //----------------------------------------------------------------------------------------------------------------------
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Creates the shader used to draw the boat model
    //----------------------------------------------------------------------------------------------------------------------
    void createShader(std::string _texture);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The shader program used to draw the boat model
    //----------------------------------------------------------------------------------------------------------------------
    ShaderProgram* m_shaderProgram;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief A vertex shader
    //----------------------------------------------------------------------------------------------------------------------
    Shader* m_vertShader;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief A fragment shader
    //----------------------------------------------------------------------------------------------------------------------
    Shader* m_fragShader;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The shader program used to draw the boat model with a clipping plane
    //----------------------------------------------------------------------------------------------------------------------
    ShaderProgram* m_shaderProgramClipped;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief A vertex shader for shading with a clipping plane
    //----------------------------------------------------------------------------------------------------------------------
    Shader* m_vertShaderClipped;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief An object used for loading the obj file into a vertex array object
    //----------------------------------------------------------------------------------------------------------------------
    ModelLoader* m_model;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief A wood texture used to texture the boat
    //----------------------------------------------------------------------------------------------------------------------
    Texture* m_tex;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief A angle of rotation to simulate a bobbing motion
    //----------------------------------------------------------------------------------------------------------------------
    float m_angle;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief A boolean to dictate the direction the boat is rotating
    //----------------------------------------------------------------------------------------------------------------------
    bool m_forwards;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The translation in the y axis for increase the bobbing effect
    //----------------------------------------------------------------------------------------------------------------------
    float m_yTrans;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief A boolean to dictate the direction the boat is translating in the y axis
    //----------------------------------------------------------------------------------------------------------------------
    bool m_upwards;
    //----------------------------------------------------------------------------------------------------------------------
};

#endif
/*@}*/
