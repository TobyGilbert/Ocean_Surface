#version 400

layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec3 vertexNormal;
layout (location = 2) in vec2 textureCoord;

out vec3 normal;
out vec4 viewVector;
out vec4 position;
out float height;

uniform vec4 cameraPosition;
out vec4 eyeVector;

uniform mat3 normalMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;

void main(){
  height = vertexPosition.y;
  position = modelViewMatrix * vec4(vertexPosition, 1.0);
  viewVector = normalize(position - cameraPosition);
  eyeVector = normalize(cameraPosition - position);
  normal = vertexNormal;
  gl_Position = modelViewProjectionMatrix * vec4(vertexPosition, 1.0);
}
