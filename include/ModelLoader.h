#ifndef MODELLOADER_H
#define MODELLOADER_H

#include "assimp/Importer.hpp"
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <vector>
#include <glm/glm.hpp>

#include <OpenGL/gl3.h>
class Mesh;
class ModelLoader{
public:
    ModelLoader(char *_fileName);
    ~ModelLoader();
    void loadMesh(const aiNode *_node, const aiScene *_scene);
    void processMesh(const aiMesh *_mesh);
    void render();
private:
    std::vector<Mesh*> m_meshes;
};

class Mesh{
public:
    Mesh(std::vector<glm::vec3> *_vertices, std::vector<glm::vec3> *_normals, std::vector<glm::vec2> *_texCoords, std::vector<GLuint> *_indices);
    ~Mesh();
    void render();
private:
    GLuint m_VAO;
    GLuint m_verticesVBO;
    GLuint m_normalsVBO;
    GLuint m_texCoordsVBO;
    GLuint m_indicesVBO;
    int m_numIndices;
};

#endif
