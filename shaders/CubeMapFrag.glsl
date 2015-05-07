#version 400
uniform samplerCube cubeMapTex;

in vec3 viewDir;
out vec4 fragColour;

uniform vec3 sunPos;

void  main(){
  vec3 rayDir = normalize(-viewDir.xyz);
  vec3 sunDir = normalize(-sunPos);
  float sunAmount = max(dot( rayDir, sunDir ), 0.0 );
  vec3 sun = mix(vec3(1.0,0.9, 0.7), vec3(1.0, 1.0, 1.0), pow(sunAmount,30.0));
  vec3 fogColour  = mix(texture(cubeMapTex, viewDir).xyz, sun, pow(sunAmount,10.0) );

  if (viewDir.y < -0.05){
    fragColour = vec4(0.1,0.19,0.22, 1.0);
  }
  else{
    fragColour =  vec4(fogColour, 1.0);//texture(cubeMapTex, viewDir);
  }
}
