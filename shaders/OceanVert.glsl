#version 400

layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec3 vertexNormal;
layout (location = 2) in vec2 textureCoord;
layout (location = 3) in vec3 offset;

out vec3 normal;
out vec2 texCoord;
out vec4 viewVector;
out vec4 position;
out float height;
out vec3 vertpos;

uniform vec4 cameraPosition;
uniform int numLayers;
out vec4 eyeVector;

uniform float time;

uniform mat3 normalMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;

uniform sampler2D perlinTexture;
uniform sampler2D fftTexture;

//vec3 calcNormals(vec2 perlinTexCoord, vec2 fftTexCoord, float dist){
//  // Calculate our vertex normals
//  vec3 norm;
//  float nL, nR, nU, nD;
//  const ivec3 off = ivec3(-1, 0, 1);
//  if (texCoord.x > 0){
//      nL = textureOffset(fftTexture, fftTexCoord, off.xy).r * 100.0;
//    //  nL += ((textureOffset(perlinTexture, perlinTexCoord, off.xy).r +  5.0) * dist);
//  }
//  else{
//      nL = 0.0;
//  }
//  if (texCoord.x < 255){
//      nR = textureOffset(fftTexture, fftTexCoord, off.zy).r * 100.0 ;
//    //  nR += ((textureOffset(perlinTexture, perlinTexCoord, off.zy).r + 5.0) * dist);

//  }
//  else{
//      nR = 0.0;
//  }
//  if (texCoord.y < 255){
//      nU = textureOffset(fftTexture, fftTexCoord, off.yz).r * 100.0;
//     // nU += ((textureOffset(perlinTexture, perlinTexCoord, off.yz).r + 5.0) * dist);

//  }
//  else{
//      nU = 0.0;
//  }
//  if (texCoord.y > 0){
//      nD = textureOffset(fftTexture, fftTexCoord,off.yx).r * 100.0;
//     // nD += ((textureOffset(perlinTexture, perlinTexCoord,off.yx).r + 5.0) * dist);
//  }
//  else{
//      nD = 0.0;
//  }
//  norm.x = nL - nR;
//  norm.y = 2.0;
//  norm.z = nD - nU;
//  norm.x = norm.x /sqrt(norm.x*norm.x + norm.y*norm.y + norm.z * norm.z);
//  norm.y = norm.y /sqrt(norm.x*norm.x + norm.y*norm.y + norm.z * norm.z);
//  norm.z = norm.z /sqrt(norm.x*norm.x + norm.y*norm.y + norm.z * norm.z);

//  return norm;
//}

out float dist;
void main(){
  vec3 pos = offset + vertexPosition;
  texCoord = ((pos.xz + ((numLayers-((numLayers/2)+1)) * 500.0) + 250.0)  / (numLayers * 500.0));// + (time/100.0);

  dist = min(1.0, max(0.0, length(vertexPosition + offset) / 1500.0));

//  pos.y = (texture(fftTexture, (vertexPosition.xz + 250)/ 500).r) * 100;
  pos.y = (vertexPosition.y * 100.0);
  vertpos = vertexPosition;

 // pos.y += (((texture(perlinTexture, texCoord).r) + 5.0)* dist);
  position = modelViewMatrix * vec4(pos, 1.0);
  viewVector = normalize(cameraPosition - position);
  eyeVector = normalize(cameraPosition - position);

  normal = normalize(normalMatrix * vertexNormal);//(normalMatrix* calcNormals(texCoord, (vertexPosition.xz + 250 )/ 500.0, dist));
  height = pos.y;
  gl_Position = modelViewProjectionMatrix * vec4(pos, 1.0);
}
