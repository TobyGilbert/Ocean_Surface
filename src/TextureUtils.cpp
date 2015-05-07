#include "TextureUtils.h"
#include <QImage>
#include <QGLWidget>
//-------------------------------------------------------------------------------------------------------------------------
GLuint TextureUtils::createTexture(const GLchar *path){
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    QImage image(path);
    QImage tex = QGLWidget::convertToGLFormat(image);
    glTexImage2D(GL_TEXTURE_2D,0, GL_RGBA, tex.width(), tex.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tex.bits());
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    return texture;

}
//-------------------------------------------------------------------------------------------------------------------------
