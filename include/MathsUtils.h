#ifndef MATHSUTILS_H
#define MATHUTILS_H
//----------------------------------------------------------------------------------------------------------------------
#include <cuda_runtime.h>
//----------------------------------------------------------------------------------------------------------------------
/// @brief Normalises a float2
/// @param _i the float2 you want to normalise
//----------------------------------------------------------------------------------------------------------------------
float2 normalise(float2 _i) {
    float length = sqrt( (_i.x*_i.x) + (_i.y*_i.y) );
    return make_float2(_i.x / length, _i.y/length);
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief Normalises a float3
/// @param _i the float3 you want to normalise
//----------------------------------------------------------------------------------------------------------------------
float3 normalise(float3 _i) {
    float length = sqrt( (_i.x*_i.x) + (_i.y*_i.y)  + (_i.z*_i.z) );
    return make_float3(_i.x/length, _i.y/length, _i.z/length);
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief Normalises a float4
/// @param _i the float4 you want to normalise
//----------------------------------------------------------------------------------------------------------------------
float4 normalise(float4 _i) {
    float length = sqrt( (_i.x*_i.x) + (_i.y*_i.y) + (_i.z*_i.z) + (_i.w*_i.w) );
    return make_float4(_i.x/length, _i.y/length, _i.z/length, _i.w/length);
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief Returns the dot product of two float2s
/// @param _x the first float2
/// @param _y the second float2
//----------------------------------------------------------------------------------------------------------------------
//float dot(float2 _x, float2 _y){
//    return (_x.x*_y.x + _x.y*_y.y);
//}
//----------------------------------------------------------------------------------------------------------------------
/// @brief Returns the length of a float2
/// @brief the float2 you want to find the length of
//----------------------------------------------------------------------------------------------------------------------
//float length(float2 _i){
//    return sqrt( (_i.x*_i.x) + (_i.y*_i.y) );
//}
//----------------------------------------------------------------------------------------------------------------------
#endif
//----------------------------------------------------------------------------------------------------------------------
