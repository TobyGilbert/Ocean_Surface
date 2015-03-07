#version 400

layout (location = 0) in vec2 vertexPosition;
layout (location = 1) in vec3 vertexNormal;
//layout (location = 2) in vec2 texCoord;
layout (location = 4) in float vertexHeight;

out vec3 position;
out vec3 normal;

//out vec2 TexCoords;

uniform mat4 modelViewMatrix;
uniform mat3 normalMatrix;
uniform mat4 projectionMatrix;
uniform mat4 modelViewProjectionMatrix;


void main(){
   //TexCoords = texCoord;
   normal = normalize(normalMatrix * vertexNormal);
   position = vec3(modelViewMatrix * vec4(vertexPosition.x, vertexHeight, vertexPosition.y, 1.0));
   gl_Position = modelViewProjectionMatrix * vec4(vertexPosition.x, vertexHeight, vertexPosition.y,1.0);
}
