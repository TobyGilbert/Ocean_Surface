#version 400

in vec4 position;
in vec4 viewVector;
in vec3 normal;
in vec4 eyeVector;
in float height;
in vec2 texCoord;
in float dist;

out vec4 fragColour;

struct sunInfo {
  float strength;
  vec3 colour;
  vec3 position;
  float shininess;
};
uniform sunInfo sun;

vec3 lightPos = vec3(0.0, 30.0, -500.0);
uniform mat3 normalMatrix;
uniform vec4 cameraPosition;

// Our cube map for generating reflections
uniform samplerCube enviroMap;

// Our perlin nosie texture generated by libnoise
uniform sampler2D perlinTexture;

uniform sampler2D fftTexture;
uniform mat4 modelViewMatrix;

vec3 skyColour(vec3 eye){
  eye.y = max(eye.y, 0.0);
  vec3 colour;
  colour.x = pow(1.0-eye.y, 2.0);
  colour.y = 1.0 - eye.y;
  colour.z = 0.6 + (1.0-eye.y) * 0.4;
  return colour;
}

float Fresnel(){
  float fresnel = 1.0 - max(dot(normalize(normal), -viewVector.xyz), 0.0);
  fresnel = pow(fresnel, 3.0) * 0.65;
  return fresnel;
}

vec3 diffuse(){
  vec3 s = normalize(sun.position - position.xyz);
  return vec3(max(dot(s,normal),0.0));
}

vec3 specular(){
   vec3 n = normalize(normalMatrix * normal);
   vec3 s = normalize(sun.position - position.xyz);
   vec3 v = normalize(vec3(-position));
   vec3 r = reflect(-s, n);
   vec3 h = normalize(v + s);
   return vec3((pow(max(dot(h, n), 0.0), 100.0)));
}
// Sun streak referenced from https://www.shadertoy.com/view/4dl3zr
vec3 sunStreak(){
  vec3 r = normalize(reflect(normalMatrix* -sun.position, normal));
  return vec3((0.8 * pow(max(0.0, dot(r, normalize(-position.xyz))), 200.0)));
}

void main(){
  vec3 N = normalize(normal);
  vec3 seaBaseCol = vec3(0.1,0.19,0.22);
  vec3 seaTopCol = vec3(0.8,0.9,0.6);

  // Calculate a reflected based on the reflection map and a refraction
  // colour based on our diffuse function
  vec4 reflectionVector = normalize(reflect(viewVector, vec4(N, 1.0)));
  vec3 reflectedCol = texture(enviroMap, reflectionVector.xyz).xyz;
  vec3 refractedCol = seaBaseCol + diffuse() * seaTopCol * 0.12;

  vec3 colour = mix(refractedCol, reflectedCol, Fresnel());
  vec3 distance = (position.xyz - cameraPosition.xyz);
  float attenuation = max(1.0 - dot(distance,distance) * 0.001, 0.0);
  colour += seaTopCol * (position.z -  0.6) * 0.18 * attenuation;
//  colour += specular();
  colour += sunStreak();

//  fragColour = vec4(vec3(dist), 1.0);
  fragColour =vec4(vec3(colour), 1.0);
//  fragColour =vec4(vec3(height/200.0), 1.0);
//  fragColour =vec4(normal, 1.0);
//  fragColour = texture(perlinTexture, texCoord);
}
