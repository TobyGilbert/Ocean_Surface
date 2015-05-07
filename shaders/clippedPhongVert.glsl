#version 400

layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec3 vertexNormal;
layout (location = 2) in vec2 texCoords;


out vec4 position;
out vec3 normal;
out vec2 TexCoords;

uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat3 normalMatrix;
uniform mat4 modelViewProjectionMatrix;

float gl_ClipDistance[1];

void main(){
   TexCoords = texCoords;
   gl_ClipDistance[0] = dot(vec4(0.0, 1.0, 0.0, -0.25), vec4(vertexPosition, 1.0));
   normal = normalize(normalMatrix * vertexNormal);
   position = modelViewMatrix * vec4(vertexPosition, 1.0);
   gl_Position = modelViewProjectionMatrix * vec4(vertexPosition,1.0);
}
