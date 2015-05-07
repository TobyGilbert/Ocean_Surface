/** @addtogroup OceanFFTStandAlone */
/*@{*/

#ifndef OCEAN_H_
#define OCEAN_H_

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
//----------------------------------------------------------------------------------------------------------------------
/// @brief A structure for storing attributes of a wave used in Gerstner's wave model
//----------------------------------------------------------------------------------------------------------------------
struct wave{
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The frequency
    //----------------------------------------------------------------------------------------------------------------------
    float W;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The amplitude
    //----------------------------------------------------------------------------------------------------------------------
    float A;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Controls the steepness of a wave
    //----------------------------------------------------------------------------------------------------------------------
    float Q;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The direction
    //----------------------------------------------------------------------------------------------------------------------
    glm::vec2 D;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The speed
    //----------------------------------------------------------------------------------------------------------------------
    float S;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief
    //----------------------------------------------------------------------------------------------------------------------
    float L;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The phase constant
    //----------------------------------------------------------------------------------------------------------------------
    float phaseConstant;
    //----------------------------------------------------------------------------------------------------------------------
};
//----------------------------------------------------------------------------------------------------------------------
/// @brief Calls the Cuda kernel for updating the height and x, z displacement of the vertices in the grid
/// @param d_position a mapped pointer to an OpenGL buffer for storing vertex positions
/// @param d_norms a mapped pointer to an OpenGL buffer for storing vertex normals
/// @param d_height the new heights calculated by our FFT kernel
/// @param d_chopX a x displacement used to reduce roundness of the peaks of waves
/// @param d_chopZ a z displacement used to reduce roundness of the peaks of waves
/// @param _choppiness a scaler used to increase the influence of the x and z displacement
/// @param _res the resolution of the ocean tile
/// @param _scale a scaler used to reduce the sclae of the heights output from FFT
//----------------------------------------------------------------------------------------------------------------------
void updateHeight(float3* d_position, float3* d_norms,  float2* d_height, float2 *d_chopX, float2 *d_chopZ, float _choppiness, int _res, float _scale);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Calls the Cuda kernel for caluclating vertex heights using the Gerstner wave model
/// @param d_heightPointer
/// @param d_normalPointer
/// @param d_waves
/// @param _time
/// @param _res
/// @param _numWaves
//----------------------------------------------------------------------------------------------------------------------
void updateGerstner(glm::vec3 *d_heightPointer,glm::vec3* d_normalPointer, wave *d_waves, float _time, int _res, int _numWaves);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Calls the Cuda kernel for calculating a field of frequency amplitudes
/// @param d_h0 An OpenGL buffer for storing the field of frequency amplitudes a time zero
/// @param d_ht An OpenGL buffer for outputting the the frequency field at time _time
/// @param _time The current time in the simulation
/// @param _res The resolution of the grid
//----------------------------------------------------------------------------------------------------------------------
void updateFrequencyDomain(float2 *d_h0, float2 *d_ht, float _time, int _res);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Adds x displacement to the wave simulation using equation 29 of Tessendorf's paper
/// @param d_Heights a mapped pointer to the heigths of the vertices calculated by FFT
/// @param d_chopX x displacement used to reduce roundness of the peaks of waves
/// @param d_chopZ a z displacement used to reduce roundness of the peaks of waves
/// @param _res The resolution of the grid
/// @param _windDirection the normalised direction of the wind
//----------------------------------------------------------------------------------------------------------------------
void addChoppiness(float2* d_Heights, float2 *d_chopX, float2 *d_chopZ, int _res, float2 _windDirection);
//----------------------------------------------------------------------------------------------------------------------
#endif
/*@}*/
