#version 400

layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec3 vertexNormal;
//layout (location = 2) in vec2 texCoord;
layout (location = 5) in vec3 vertexColours;

out vec4 position;
out vec3 normal;
out vec3 colour;
out float height;
out vec2 texCoords;

//out vec2 TexCoords;

uniform mat4 modelViewMatrix;
uniform mat3 normalMatrix;
uniform mat4 projectionMatrix;
uniform mat4 modelViewProjectionMatrix;


void main(){
   texCoords = vec2(((vertexPosition.x+500.0)/1000), ((vertexPosition.z+500)/1000));
   //TexCoords = texCoord;
   height = vertexPosition.y;
   colour = vertexColours;
   normal = normalize(normalMatrix * vertexNormal);
   position = modelViewProjectionMatrix * vec4(vertexPosition, 1.0);
   gl_Position = modelViewProjectionMatrix * vec4(vertexPosition,1.0);
}
