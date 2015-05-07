  #version 400

in vec4 position;
in vec4 viewVector;
in vec3 normal;
in vec4 eyeVector;
in float height;
in vec2 texCoord;
in float dist;
in vec3 vertpos;
in vec4 MVPPos;
in vec2 texCoordTile;
in vec3 vertPos;

out vec4 fragColour;

struct sunInfo {
  vec3 position;
  float streak;
};
//----------------------------------
//-------------Uniforms-------------
//----------------------------------
uniform sunInfo sun;
uniform float time;
uniform mat3 normalMatrix;
uniform vec4 cameraPosition;
uniform vec3 seaBaseColour;
uniform vec3 seaTopColour;
uniform sampler2D reflectTex;
uniform sampler2D dudvTex;
uniform float fogFarDist;
uniform float fogNearDist;
uniform mat4 modelViewMatrix;

float Fresnel(){
    float f = 1.0 - max(dot(normal, -viewVector.xyz), 0.0);
    f = pow(f, 2.0) * 0.65;
    return f;
}

float diffuse(){
  vec3 s = normalize(sun.position - position.xyz);
  return max(dot(s,normal),0.0);
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
  vec3 r = normalize(reflect((normalMatrix*-sun.position), normal));
  return vec3((1.0 * pow(max(0.0, dot(r, normalize(-position.xyz))), sun.streak))) * vec3(1.0,0.9, 0.7);
}

// Constants //
float kDistortion = 0.015;
float kReflection = 0.01;

void main(){
  vec3 N = normalize(normal);

  // Calculate a reflected based on the reflection map and a refraction
//  vec3 reflectionVector = viewVector.xyz - 2.0 * normal * dot(normal, viewVector.xyz);

  vec4 distOffset = texture(dudvTex, texCoord + vec2(time/100)) * kDistortion;
  vec4 dudvColour = texture(dudvTex, vec2(texCoord + distOffset.xy));
  dudvColour = normalize(dudvColour * 2.0 -1.0) * kReflection;

  vec4 tmp = vec4(1.0/ MVPPos.w);
  vec4 projCoord = MVPPos * tmp;
  projCoord += vec4(1.0);
  projCoord *= vec4(0.5);
  projCoord += dudvColour;
  projCoord = clamp(projCoord, 0.001, 0.999);

  vec3 reflectedCol = texture(reflectTex, projCoord.xy).xyz;
  vec3 refractedCol = seaBaseColour + vec3(diffuse()) * seaTopColour * 0.12;

  vec3 colour = mix(refractedCol, reflectedCol, Fresnel());
  vec3 distance = (position.xyz - cameraPosition.xyz);
  float attenuation = max(1.0 - dot(distance,distance) * 0.01, 0.0);
  colour += seaTopColour * (position.z -  0.6) * 0.18 * attenuation;
  colour += sunStreak();

  // Fog
//  float fogFactor = (fogFarDist - length(vertPos)) / (fogFarDist - fogNearDist);
//  fogFactor = clamp(fogFactor, 0.0, 1.0);

  fragColour = vec4(colour, 1.0);

}
