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
out vec4 MVPPos;
out vec2 texCoordTile;
out vec3 vertPos;

uniform vec4 cameraPosition;
uniform int numLayers;
out vec4 eyeVector;

uniform float time;

uniform mat3 normalMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;

out float dist;

void main(){
  vertPos = offset +  vertexPosition;
  texCoordTile = textureCoord;
  vec3 pos = offset + vertexPosition;
  texCoord = ((pos.xz + ((numLayers-((numLayers/2)+1)) * 500.0) + 250.0)  / (numLayers * 500.0));// + (time/100.0);

  dist = min(1.0, max(0.0, length(vertexPosition + offset) / 1500.0));

  pos.y = (vertexPosition.y * 100.0);
  vertpos = vertexPosition;

  position = modelViewMatrix * vec4(pos, 1.0);
  viewVector = normalize(position - cameraPosition);
  eyeVector = normalize(cameraPosition - position);

  normal = normalize(normalMatrix * vertexNormal);
  height = pos.y;
  gl_Position = modelViewProjectionMatrix * vec4(pos, 1.0);
  MVPPos = modelViewProjectionMatrix * vec4(pos, 1.0);
}
