#version 400

in vec4 position;
in vec4 viewVector;
in vec3 normal;
in vec4 eyeVector;
in float height;

out vec4 fragColour;

struct sunInfo {
  float strength;
  vec3 colour;
  vec3 direction;
  float shininess;
};

uniform sunInfo sun;
uniform mat3 normalMatrix;
uniform vec4 cameraPosition;

uniform samplerCube enviroMap;

float Fresnel(float NdotL, float fresnelBias, float fresnelPow){
  float facing = (1.0 - NdotL);
  return min(1.0,max(fresnelBias + (1.0 - fresnelBias) * pow(facing, fresnelPow), 0.0));
}

void main(){
  vec3 N = normalize(normalMatrix * normal);
  vec4 reflectionVector = normalize(reflect(viewVector, vec4(N, 1.0)));
  float fresnel = Fresnel(dot(N, normalize(sun.direction)), 0.01, 4.0);
  vec4 globalReflect = texture(enviroMap, reflectionVector.xyz);
  vec4 sunReflect = vec4(sun.strength) * vec4(sun.colour,1.0) * pow(max(0.0, min(dot(reflectionVector, normalize(vec4(sun.direction, 1.0))),1.0)), sun.shininess);
  vec4 refraction = (vec4(2.0) * vec4(107.0/255.0, 138.0/255.0, 159.0/255.0, 1.0) * vec4(height + 50.0) )/ 160.0;
  fragColour = vec4((1-fresnel)) * refraction + vec4(fresnel) * (globalReflect + sunReflect);
}
