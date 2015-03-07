#include "ModelLoader.h"
#include "iostream"

ModelLoader::ModelLoader(char *_fileName){
    Assimp::Importer import;
    const aiScene* scene = import.ReadFile(_fileName, aiProcess_GenSmoothNormals | aiProcess_Triangulate );
    if(scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        std::cerr<<"The file was not successfully opened: "<<_fileName<<std::endl;
        return;
    }
    loadMesh(scene->mRootNode, scene);
}

ModelLoader::~ModelLoader(){
    for (int i=0; i<m_meshes.size(); i++){
        delete m_meshes[i];
    }
    m_meshes.clear();
}

void ModelLoader::loadMesh(const aiNode* _node, const aiScene* _scene){
    for(int i = 0; i < _node->mNumMeshes; i++){
        aiMesh* mesh = _scene->mMeshes[_node->mMeshes[i]];
        processMesh(mesh);
    }

    for(int i = 0; i < _node->mNumChildren; i++){
        loadMesh(_node->mChildren[i], _scene);
    }
}

void ModelLoader::processMesh(const aiMesh* _mesh){
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texCoords;
    std::vector<GLuint> indices;

    for(int i = 0; i <_mesh->mNumVertices; i++)
    {
        glm::vec3 tempVec;

        // position
        tempVec.x = _mesh->mVertices[i].x;
        tempVec.y = _mesh->mVertices[i].y;
        tempVec.z = _mesh->mVertices[i].z;

        vertices.push_back(tempVec);

        // normals
        tempVec.x = _mesh->mNormals[i].x;
        tempVec.y = _mesh->mNormals[i].y;
        tempVec.z = _mesh->mNormals[i].z;

        normals.push_back(tempVec);

        // UV
        if(_mesh->mTextureCoords[0]){
            tempVec.x = _mesh->mTextureCoords[0][i].x;
            tempVec.y = _mesh->mTextureCoords[0][i].y;
        }
        else{
            tempVec.x = tempVec.y = 0.0;
        }

        texCoords.push_back(glm::vec2(tempVec.x, tempVec.y));
    }

    for(int i = 0; i < _mesh->mNumFaces; i++){
        aiFace face = _mesh->mFaces[i];

        for(int j = 0; j < face.mNumIndices; j++){
            indices.push_back(face.mIndices[j]);
        }
    }

    Mesh* mesh = new Mesh(&vertices, &normals, &texCoords, &indices);
    m_meshes.push_back(mesh);
}

void ModelLoader::render(){
    for (int i=0; i<m_meshes.size(); i++){
        m_meshes[i]->render();
    }
}

Mesh::Mesh(std::vector<glm::vec3> *_vertices, std::vector<glm::vec3> *_normals, std::vector<glm::vec2> *_texCoords, std::vector<GLuint> *_indices){
    m_numIndices = _indices->size();
    std::vector <glm::vec3> verts = *_vertices;
    std::vector <glm::vec3> norms = *_normals;
    std::vector <glm::vec2> tex = *_texCoords;
    std::vector <GLuint> ind = *_indices;

    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    glGenBuffers(1, &m_verticesVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_verticesVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*_vertices->size(), &verts[0].x, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glGenBuffers(1, &m_normalsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_normalsVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*_normals->size(), &norms[0].x, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glGenBuffers(1, &m_texCoordsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_texCoordsVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2)*_texCoords->size(), &tex[0].x, GL_STATIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glGenBuffers(1, &m_indicesVBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indicesVBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*_indices->size(), &ind[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

}

Mesh::~Mesh(){
    glDeleteBuffers(1, &m_verticesVBO);
    glDeleteBuffers(1, &m_normalsVBO);
    glDeleteBuffers(1, &m_texCoordsVBO);
    glDeleteBuffers(1, &m_indicesVBO);

    glDeleteVertexArrays(1, &m_VAO);
}

void Mesh::render(){
    glBindVertexArray(m_VAO);
//    glDrawElements(GL_TRIANGLES, m_numIndices, GL_UNSIGNED_INT, 0);
    glDrawArrays(GL_TRIANGLES, 0, m_numIndices);
    glBindVertexArray(0);
}
