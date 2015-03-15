//#version 400

//in vec3 position;
//in vec3 normal;
////in vec2 TexCoords;

//struct lightInfo{
//   vec4 position;
//   vec3 intensity;
//};

//uniform lightInfo light;

//uniform vec3 Kd;
//uniform vec3 Ka;
//uniform vec3 Ks;
//uniform float shininess;
//uniform sampler2D tex;
//uniform vec3 cameraPos;

//out vec4 fragColour;
//in vec3 colour;
//in float height;

//float fresnel(){
//   vec3 I; // vector from eye to point;
//   I = cameraPos- position;
//   float bias = 0.0;
//   float scale = 0.001;
//   float power = 1.0;
//   return max(0.0, min(1.0, bias + scale * (float(1.0 + dot(I, normalize(normal))))));
//}

//vec3 ads(){
//   vec3 n = normalize(normal);
//   vec3 s = normalize(vec3(light.position) - position);
//   vec3 v = normalize(vec3(-position));
//   vec3 r = reflect(-s, n);
//   vec3 h = normalize(v + s);
//   return light.intensity * (Ka + Kd * max(dot(s,n),0.0)+ Ks * pow(max(dot(h, n), 0.0), shininess));
//}

//void  main(){
////   fragColour = vec4(vec3(((height/10.0) + 10)/20),1.0);//* vec4(0.0, 0.0, 1.0, 1.0);
////  fragColour = vec4(vec3(fresnel()), 1.0);// * vec4(3.0/255.0, 109.0/255.0, 141.0/255.0, 1.0);
//  vec3 skyColour = vec3(2.0/255.0, 151.0/255.0, 250.0/255.0);
//  vec3 waterColour = vec3(3.0/255.0, 109.0/255.0, 141.0/255.0);
//  vec3 skyAmbient = vec3(1.0, 0.0, 0.0);
//  float f = fresnel();
//  vec3 x = vec3(f) ;//∗ vec3(skyColour);// + vec3(1.0 − f);// ∗ waterColour ∗ skyColour ∗ skyAmbient;
//  x *= skyColour;
//  x += vec3(1.0 - f) * waterColour * skyColour * skyAmbient;
//  fragColour = vec4(x, 1.0);
//}
#version 400
uniform vec3 cameraPos;
out vec4 fragColour;

in float height;

in vec2 texCoords;
in vec4 position;
struct lightInfo{
   vec4 position;
   vec3 intensity;
};

uniform lightInfo light;

uniform vec3 Kd;
uniform vec3 Ka;
uniform vec3 Ks;
uniform float shininess;
vec3 LightDirection = -light.position.xyz;

//vec3 reflectMap = vec3(2.0/255.0, 151.0/255.0, 250.0/255.0);
uniform sampler2D reflectMap;
vec3 refractMap = vec3(3.0/255.0, 109.0/255.0, 141.0/255.0);

in vec3 normal;

// Constants //
float kDistortion = 0.015;
float kReflection = 0.01;
vec4 baseColour = vec4(19.0/255.0, 38.0/255.0 ,51.0/255.0, 1.0);

vec4 tangent = vec4(1.0, 0.0, 0.0, 0.0);
vec4 lightNormal = vec4(0.0, 1.0, 0.0, 0.0);
vec4 biTangent = vec4(0.0, 0.0, 1.0, 0.0);
vec3 ads(){
   vec3 n = normalize(normal);
   vec3 s = normalize(vec3(light.position) - position.xyz);
   vec3 v = normalize(vec3(-position));
   vec3 r = reflect(-s, n);
   vec3 h = normalize(v + s);
   return light.intensity * (Ka + Kd * max(dot(s,n),0.0)+ Ks * pow(max(dot(h, n), 0.0), shininess));
}
float fresnel(){
   vec3 I; // vector from eye to point;
   I = cameraPos - position.xyz;
   float bias = 0.0;
   float scale = 0.005;
   float power = 1.0;
   return max(0.0, min(1.0, bias + scale * (float(1.0 + dot(I, normalize(normal))))));
}
void main(){
  // Light tangent space
  vec4 lightDir = normalize(vec4(LightDirection.xyz, 1.0));
  vec4 lightTanSpace = normalize(vec4(dot(lightDir, tangent), dot(lightDir, biTangent), dot(lightDir, lightNormal), 1.0));

  // Fresnal Term
//  vec4 distOffset = texture(dudv, TexCoords + vec2(time)) * kDistortion;
//  vec4 normal = texture(normalMap, vec2(TexCoords + distOffset.xy));
  //  normal = normalize(normal * 2.0 - 1.0);
  //  normal.a = 0.81;
  vec4 norm = vec4(normalize(normal * 2.0 - 1.0), 0.81);

  vec4 lightReflection = normalize(reflect(-1 * lightTanSpace, norm));
//  vec4 invertedFresnal = vec4(dot(norm, lightReflection));
//   vec4 fresnalTerm = 1.0 - invertedFresnal;
  vec4 fresnalTerm = vec4(fresnel());
  vec4 invertedFresnal = vec4(1.0) - fresnalTerm;

  // Reflection
//  vec4 dudvColour = texture(dudv, vec2(TexCoords + distOffset.xy));
//  dudvColour = normalize(dudvColour *2.0 -1.0) * kReflection;

  // Projection coordinates from http://www.bonzaisoftware.com/tnp/gl-water-tutorial/
  vec4 tmp = vec4(1.0/ position.w);
  vec4 projCoord = position * tmp;
  projCoord += vec4(1.0);
  projCoord *= vec4(0.5);
//  projCoord += dudvColour;
  projCoord = clamp(projCoord, 0.001, 0.999);

  vec4 reflectionColour = mix(texture(reflectMap, projCoord.xy), baseColour, 0.3);
  reflectionColour *= fresnalTerm;

  // Refraction
  vec4 refractionColour = vec4(refractMap, 1.0);
  vec4 depthValue = vec4(0.1, 0.1, 0.1, 1.0);
  vec4 invDepth = 1.0 - depthValue;

  refractionColour *= invertedFresnal * invDepth;
  refractionColour += baseColour * depthValue * invertedFresnal;

  float heightScale = (height + 24) /15;

  fragColour = reflectionColour + refractionColour* vec4(heightScale);
}


//void main(){
//  vec3 upwelling = vec3(0.0, 0.2, 0.3);
//  vec3 sky = vec3(0.69,0.84,1.0);
//  vec3 air = vec3(0.1,0.1,0.1);
//  float nSnell = 1.34;
//  float Kdiffuse = 0.91;
//  float reflectivity;
//  vec3 I = position.xyz - light.position.xyz;
//  vec3 nI = normalize(I);
//  vec3 nN = normalize(normal);
//  float costhetai = abs(dot(nI, nN));
//  float thetai = acos(costhetai);
//  float sinthetai = sin(thetai / nSnell);
//  float thetat = asin(sinthetai);
//  if (thetai == 0.0){
//      reflectivity = (nSnell - 1) / (nSnell + 1);
//      reflectivity = reflectivity * reflectivity;
//  }
//  else{
//      float fs = sin(thetat - thetai) / sin(thetat + thetai);
//      float ts = tan(thetat - thetai) / tan(thetat + thetai);
//      reflectivity = 0.5 * (fs * fs + ts * ts);
//  }
//  vec3 dPE = position.xyz - cameraPos;
//  float dist = length(dPE) * Kdiffuse;
////  dist = exp(-dist);
//  vec3 colour = dist * (reflectivity * sky  + (1-reflectivity) * upwelling) + (1-dist) * air;
//  fragColour = vec4(colour, 1.0);
//}
