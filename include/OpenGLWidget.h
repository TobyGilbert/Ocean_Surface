#ifndef OPENGLWIDGET_H
#define OPENGLWIDGET_H



#include <QGLWidget>
#include <QEvent>
#include <QResizeEvent>
#include <QMessageBox>
#include <Camera.h>
#include <glm/glm.hpp>

#include "ShaderProgram.h"
#include "Shader.h"
#include "ModelLoader.h"
#include "OceanGrid.h"
#include "Skybox.h"


class OpenGLWidget : public QGLWidget
{
    Q_OBJECT //must include to gain access to qt stuff

public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief ctor for our NGL drawing class
    /// @param [in] parent the parent window to the class
    //----------------------------------------------------------------------------------------------------------------------
    explicit OpenGLWidget(const QGLFormat _format, QWidget *_parent=0);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief dtor must close down and release OpenGL resources
    //----------------------------------------------------------------------------------------------------------------------
    ~OpenGLWidget();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the virtual initialize class is called once when the window is created and we have a valid GL context
    /// use this to setup any default GL stuff
    //----------------------------------------------------------------------------------------------------------------------
    void initializeGL();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this is called everytime we want to draw the scene
    //----------------------------------------------------------------------------------------------------------------------
    void paintGL();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief called to resize the window
    //----------------------------------------------------------------------------------------------------------------------
    void resizeGL(const int _w, const int _h );
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mouse move
    //----------------------------------------------------------------------------------------------------------------------
    void mouseMoveEvent(QMouseEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a timer event function from the Q_object
    //----------------------------------------------------------------------------------------------------------------------
    void timerEvent(QTimerEvent *);//
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Trigger when mouse button pressed
    //----------------------------------------------------------------------------------------------------------------------
    void mousePressEvent(QMouseEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Triggered when mosue button released
    //----------------------------------------------------------------------------------------------------------------------
    void mouseReleaseEvent(QMouseEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Triggered when a key press occurs
    //----------------------------------------------------------------------------------------------------------------------
    void keyPressEvent(QKeyEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Triggered when scroll wheel moved
    //----------------------------------------------------------------------------------------------------------------------
    void wheelEvent(QWheelEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    void testing(std::string _st);
    void genFBOs();
    void renderReflections();
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our Camera
    //----------------------------------------------------------------------------------------------------------------------
    Camera *m_cam;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Model matrix
    //----------------------------------------------------------------------------------------------------------------------
    glm::mat4 m_modelMatrix;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Height of the window
    //----------------------------------------------------------------------------------------------------------------------
    int m_height;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Width of the window
    //----------------------------------------------------------------------------------------------------------------------
    int m_width;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Mouse transforms
    //----------------------------------------------------------------------------------------------------------------------
    glm::mat4 m_mouseGlobalTX;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief model pos
    //----------------------------------------------------------------------------------------------------------------------
    glm::vec3 m_modelPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Spin face x
    //----------------------------------------------------------------------------------------------------------------------
    float m_spinXFace;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sping face y
    //----------------------------------------------------------------------------------------------------------------------
    float m_spinYFace;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief rotate bool
    //----------------------------------------------------------------------------------------------------------------------
    bool m_rotate;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief translate bool
    //----------------------------------------------------------------------------------------------------------------------
    bool m_translate;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief
    //----------------------------------------------------------------------------------------------------------------------
    int m_origX;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief
    //----------------------------------------------------------------------------------------------------------------------
    int m_origY;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief
    //----------------------------------------------------------------------------------------------------------------------
    int m_origXPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief
    //----------------------------------------------------------------------------------------------------------------------
    int m_origYPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The ocean grid
    //----------------------------------------------------------------------------------------------------------------------
    OceanGrid *m_oceanGrid;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Skybox Model
    //----------------------------------------------------------------------------------------------------------------------
    Skybox *m_skybox;
    //----------------------------------------------------------------------------------------------------------------------

    // Framebuffer
    GLuint m_reflectFBO;
};

#endif // OPENGLWIDGET_H
