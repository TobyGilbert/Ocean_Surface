#version 400
uniform samplerCube cubeMapTex;

in vec3 viewDir;
out vec4 fragColour;

void  main(){
   fragColour = texture(cubeMapTex, viewDir);
}
