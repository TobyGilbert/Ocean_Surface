#ifndef OCEAN_H_
#define OCEAN_H_
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @author Toby Gilbert
// ----------------------------------------------------------------------------------------------------------------------------------------
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief A structure for storing attributes of a wave used in Gerstner's wave model
// ----------------------------------------------------------------------------------------------------------------------------------------
struct wave{
    // ----------------------------------------------------------------------------------------------------------------------------------------
    /// @brief The frequency
    // ----------------------------------------------------------------------------------------------------------------------------------------
    float W;
    // ----------------------------------------------------------------------------------------------------------------------------------------
    /// @brief The amplitude
    // ----------------------------------------------------------------------------------------------------------------------------------------
    float A;
    // ----------------------------------------------------------------------------------------------------------------------------------------
    /// @brief Controls the steepness of a wave
    // ----------------------------------------------------------------------------------------------------------------------------------------
    float Q;
    // ----------------------------------------------------------------------------------------------------------------------------------------
    /// @brief The direction
    // ----------------------------------------------------------------------------------------------------------------------------------------
    glm::vec2 D;
    // ----------------------------------------------------------------------------------------------------------------------------------------
    /// @brief The speed
    // ----------------------------------------------------------------------------------------------------------------------------------------
    float S;
    // ----------------------------------------------------------------------------------------------------------------------------------------
    /// @brief
    // ----------------------------------------------------------------------------------------------------------------------------------------
    float L;
    // ----------------------------------------------------------------------------------------------------------------------------------------
    /// @brief The phase constant
    // ----------------------------------------------------------------------------------------------------------------------------------------
    float phaseConstant;
    // ----------------------------------------------------------------------------------------------------------------------------------------
};
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief Calls the Cuda kernel for updating the height and x, z displacement of the vertices in the grid
/// @param d_position An OpenGL buffer for storing the current positions of the vertices in the grid
/// @param d_height An OpenGL buffer which holds the new heights of grid positions
/// @param d_normal An OpenGL buffer which holds the normals
/// @param d_xDisplacement An OpenGL buffer for storing the displacment in the x axis
/// @param _res The resolution of the grid
/// @param _scale Scales the amplitude of the waves
// ----------------------------------------------------------------------------------------------------------------------------------------
void updateHeight(float3* d_position, float3* d_norms,  float2* d_height, float2 *d_chopX, float2 *d_chopZ, float _choppiness, int _res, float _scale);
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief Calls the Cuda kernel for caluclating vertex heights using the Gerstner wave model
/// @param d_point
/// @param d_point2
/// @param d_waves An OpenGL buffer for storeing waves used in the simulation
/// @param _time The current time of the simulation
/// @param _res The resolution of the grid
/// @param _numWaves The number of waves in the simulation
// ----------------------------------------------------------------------------------------------------------------------------------------
void updateGerstner(glm::vec3 *d_heightPointer,glm::vec3* d_normalPointer, wave *d_waves, float _time, int _res, int _numWaves);
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief Calls the Cuda kernel for calculating a field of frequency amplitudes
/// @param d_h0 An OpenGL buffer for storing the field of frequency amplitudes a time zero
/// @param d_ht An OpenGL buffer for outputting the the frequency field at time _time
/// @param _time The current time in the simulation
/// @param _res The resolution of the grid
// ----------------------------------------------------------------------------------------------------------------------------------------
void updateFrequencyDomain(float2 *d_h0, float2 *d_ht, float _time, int _res);
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief Adds x displacement to the wave simulation using equation 29 of Tessendorf's paper
/// @param d_ht An OpenGL buffer which holds the frequency field
/// @brief _res The resolution of the grid
// ----------------------------------------------------------------------------------------------------------------------------------------
void addChoppiness(float2* d_Heights, float2 *d_chopX, float2 *d_chopZ, int _res, float2 _windSpeed);
// ----------------------------------------------------------------------------------------------------------------------------------------
#endif
