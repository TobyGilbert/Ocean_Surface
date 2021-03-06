#version 400

in vec4 position;
in vec3 normal;
in vec2 TexCoords;

struct lightInfo{
   vec4 position;
   vec3 intensity;
};

uniform lightInfo light;

uniform vec3 Kd;
uniform vec3 Ka;
uniform vec3 Ks;
uniform float shininess;

uniform sampler2D tex;
uniform mat4 modelViewMatrix;

out vec4 fragColour;

vec3 ads(){
   vec3 n = normalize(normal);
   vec3 s = normalize((modelViewMatrix * light.position) - position).xyz;
   vec3 v = normalize(vec3(-position));
   vec3 r = reflect(-s, n);
   vec3 h = normalize(v + s);
   return light.intensity * (Ka + Kd * max(dot(s,n),0.0)+ Ks * pow(max(dot(h, n), 0.0), shininess));
}

void  main(){
  fragColour = vec4(ads()* texture(tex, TexCoords).xyz, 1.0);
}


