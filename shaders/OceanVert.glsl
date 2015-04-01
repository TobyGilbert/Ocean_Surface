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

uniform vec4 cameraPosition;
uniform int numLayers;
out vec4 eyeVector;

uniform float time;

uniform mat3 normalMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;

uniform sampler2D perlinTexture;

void main(){
  // Find the position before matrix transform
  vec3 pos = offset + vertexPosition;
  // Find the position after offseting the instance
  position = modelViewMatrix * vec4(pos, 1.0);
  // Calculate the texture coordinate based on the position of the tile
  texCoord = ((pos.xz + ((numLayers-((numLayers/2)+1)) * 500.0) + 250.0)  / (numLayers * 500.0)) + (time/100.0);
  // Add the perlin noise to the height of the waves
  position.y *= (texture(perlinTexture, texCoord).r + texture(perlinTexture, texCoord).g + texture(perlinTexture, texCoord).b) /3.0;
  pos.y *= (texture(perlinTexture, texCoord).r + texture(perlinTexture, texCoord).g + texture(perlinTexture, texCoord).b) /3.0;
  viewVector = normalize(position - cameraPosition);
  eyeVector = normalize(cameraPosition - position);
  normal = (normalMatrix * vertexNormal);
  height = pos.y;
  gl_Position = modelViewProjectionMatrix * vec4(pos, 1.0);
}
