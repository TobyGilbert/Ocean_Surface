#include "OpenGLWidget.h"
#include <QGuiApplication>
#include <QColorDialog>
#include <iostream>
#include <glm/gtc/matrix_inverse.hpp>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <QTimer>
//-------------------------------------------------------------------------------------------------------------------------
const static float INCREMENT= 2.0;
const static float ZOOM = 50;
OpenGLWidget::OpenGLWidget(const QGLFormat _format, QWidget *_parent) : QGLWidget(_format,_parent){
    // set this widget to have the initial keyboard focus
    setFocus();
    setFocusPolicy(Qt::StrongFocus);
    // re-size the widget to that of the parent (in this case the GLFrame passed in on construction)
    m_rotate=false;
    // mouse rotation values set to 0
    m_spinXFace=0;
    m_spinYFace=0;
    // re-size the widget to that of the parent (in this case the GLFrame passed in on construction)
    this->resize(_parent->size());

    m_sunPos = glm::vec3(0.0, 10.0, -50.0);
}
//----------------------------------------------------------------------------------------------------------------------
OpenGLWidget::~OpenGLWidget(){
    delete m_cam;
    delete m_oceanGrid;
    delete m_skybox;
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::initializeGL(){
#ifdef LINUX
    glewExperimental = GL_TRUE;
    GLenum error = glewInit();
    if(error != GLEW_OK){
        std::cerr<<"GLEW IS NOT OK!!!"<<std::endl;
    }
#endif

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    // enable depth testing for drawing
    glEnable(GL_DEPTH_TEST);
    // enable multisampling for smoother drawing
    glEnable(GL_MULTISAMPLE);
    // enable so we can use clipping in the shader
    glEnable(GL_CLIP_DISTANCE0);


    // as re-size is not explicitly called we need to do this.
    glViewport(0,0,width(),height());

    // Initialise the model matrix
    m_modelMatrix = glm::mat4(1.0);

    // Initialize the camera
    m_cam = new Camera(glm::vec3(0.0, 100.0, 100.0));

    m_oceanGrid = new OceanGrid(128, 500, 500);

    m_skybox = new Skybox();
    m_renderSkyBox = true;

    m_boat = new Boat("models/boat.obj", "textures/boat.jpg");
    m_renderBoat = false;

    genFBOs();

    QTimer *timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(updateGL()));
    timer->start(1000/30); // Update every 1/30sec

}
//-------------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::genFBOs(){
    // Gen framebuffer
    glGenFramebuffers(1, &m_reflectFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_reflectFBO);

    // Bind texture to FBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *(m_oceanGrid->getReflectTex()), 0);

    // Set the targets for the fragment shader output
    GLenum drawBufs[] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, drawBufs);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::resizeGL(const int _w, const int _h){
    // set the viewport for openGL
    glViewport(0,0,_w,_h);
    m_cam->setShape(_w, _h);
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::renderReflections(){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, 512, 512);

    glm::mat4 rotx;
    glm::mat4 roty;

    rotx = glm::rotate(rotx, float(m_spinXFace/20.0), glm::vec3(1.0, 0.0, 0.0));
    roty = glm::rotate(roty, float(m_spinYFace/20.0), glm::vec3(0.0, 1.0, 0.0));

    m_mouseGlobalTX = rotx*roty;
    // add the translations
    m_mouseGlobalTX[3][0] = m_modelPos.x;
    m_mouseGlobalTX[3][1] = m_modelPos.y;
    m_mouseGlobalTX[3][2] = m_modelPos.z;
    m_modelMatrix = m_mouseGlobalTX;

    m_modelMatrix = glm::scale(m_modelMatrix, glm::vec3(10000.0, -10000.0, 10000.0));
    m_skybox->loadMatricesToShader(m_modelMatrix, m_cam->getViewMatrix(), m_cam->getProjectionMatrix());
    m_skybox->render();

    if (m_renderBoat){
        m_modelMatrix= m_mouseGlobalTX;
        m_modelMatrix = glm::scale(m_modelMatrix, glm::vec3(5.0, -5.0,5.0));
        m_modelMatrix = glm::translate(m_modelMatrix, glm::vec3(0.0, 3.0, 0.0));
        m_boat->loadMatricesToShaderClipped(m_modelMatrix, m_cam->getViewMatrix(), m_cam->getProjectionMatrix());
        m_boat->render();
    }
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::paintGL(){

    // Set the sun Position
    m_oceanGrid->setSunPos(m_sunPos);
    m_skybox->setSunPos(m_sunPos);

    updateTimer(m_oceanGrid->getTime());

    glBindFramebuffer(GL_FRAMEBUFFER, m_reflectFBO);
    renderReflections();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glViewport(0, 0, width()*devicePixelRatio(), height()*devicePixelRatio());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 rotx;
    glm::mat4 roty;

    rotx = glm::rotate(rotx, float(m_spinXFace/20.0), glm::vec3(1.0, 0.0, 0.0));
    roty = glm::rotate(roty, float(m_spinYFace/20.0), glm::vec3(0.0, 1.0, 0.0));

    m_mouseGlobalTX = rotx*roty;
    m_mouseGlobalTX[3][0] = m_modelPos.x;
    m_mouseGlobalTX[3][1] = m_modelPos.y;
    m_mouseGlobalTX[3][2] = m_modelPos.z;
    m_modelMatrix = m_mouseGlobalTX;

    if (m_renderSkyBox){
        m_modelMatrix = glm::scale(m_modelMatrix, glm::vec3(10000.0, 10000.0, 10000.0));
        m_skybox->loadMatricesToShader(m_modelMatrix, m_cam->getViewMatrix(), m_cam->getProjectionMatrix());
        m_skybox->render();
    }

    m_modelMatrix = m_mouseGlobalTX;
    m_modelMatrix = glm::translate(m_modelMatrix, glm::vec3(0.0, -20.0, 0.0));
    m_oceanGrid->loadMatricesToShader(m_modelMatrix, m_cam->getViewMatrix(), m_cam->getProjectionMatrix());
    m_oceanGrid->render();

    if(m_renderBoat){
        m_modelMatrix= m_mouseGlobalTX;
        m_modelMatrix = glm::scale(m_modelMatrix, glm::vec3(5.0, 5.0, 5.0));
        m_modelMatrix = glm::translate(m_modelMatrix, glm::vec3(0.0, -3.0, 0.0));
        m_boat->loadMatricesToShader(m_modelMatrix, m_cam->getViewMatrix(), m_cam->getProjectionMatrix());
        m_boat->render();
    }
}
//------------------------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::keyPressEvent(QKeyEvent *_event){
    switch(_event->key()){
        case Qt::Key_Escape:
            QGuiApplication::exit();
            break;
        case Qt::Key_Left:
           // m_oceanGrid->moveSunLeft();
            m_sunPos.x -= 5.0;
            break;
        case Qt::Key_Right:
            //m_oceanGrid->moveSunRight();
            m_sunPos.x += 5.0;
            break;
        case Qt::Key_Up:
            //m_oceanGrid->moveSunUp();
            m_sunPos.y += 5.0;
            break;
        case Qt::Key_Down:
            //m_oceanGrid->moveSunDown();
            m_sunPos.y -= 5.0;
            break;
    }
}
//------------------------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::mouseMoveEvent (QMouseEvent *_event){
  // Sourced from Jon Macey's NGL library
  // note the method buttons() is the button state when event was called
  // this is different from button() which is used to check which button was
  // pressed when the mousePress/Release event is generated
  if(m_rotate && _event->buttons() == Qt::LeftButton){
    int diffx=_event->x()-m_origX;
    int diffy=_event->y()-m_origY;
    m_spinXFace += (float) 0.5f * diffy;
    m_spinYFace += (float) 0.5f * diffx;
    m_origX = _event->x();
    m_origY = _event->y();
  }
        // right mouse translate code
  else if(m_translate && _event->buttons() == Qt::RightButton){
    int diffX = (int)(_event->x() - m_origXPos);
    int diffY = (int)(_event->y() - m_origYPos);
    m_origXPos=_event->x();
    m_origYPos=_event->y();
    m_modelPos.x += INCREMENT * diffX;
    m_modelPos.y -= INCREMENT * diffY;
   }
}
//------------------------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::mousePressEvent ( QMouseEvent * _event){
    // Sourced from Jon Macey's NGL library
    // this method is called when the mouse button is pressed in this case we
    // store the value where the mouse was clicked (x,y) and set the Rotate flag to true
    if(_event->button() == Qt::LeftButton)
    {
        m_origX = _event->x();
        m_origY = _event->y();
        m_rotate = true;
    }
    // right mouse translate mode
    else if(_event->button() == Qt::RightButton)
    {
        m_origXPos = _event->x();
        m_origYPos = _event->y();
        m_translate = true;
    }
}
//------------------------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::mouseReleaseEvent ( QMouseEvent * _event ){
    // Sourced from Jon Macey's NGL library
  // this event is called when the mouse button is released
  // we then set Rotate to false
  if (_event->button() == Qt::LeftButton)
  {
    m_rotate=false;
  }
        // right mouse translate mode
  if (_event->button() == Qt::RightButton)
  {
    m_translate=false;
  }
}
//------------------------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::wheelEvent(QWheelEvent *_event){
    // Sourced from Jon Macey's NGL library
    // check the diff of the wheel position (0 means no change)
    if(_event->delta() > 0)
    {
        m_modelPos.z+=ZOOM;
    }
    else if(_event->delta() <0 )
    {
        m_modelPos.z-=ZOOM;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::updateChoppiness(int value){
    m_oceanGrid->updateChoppiness((float)value/100.0);
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::updateWindDirX(double _x){
    m_oceanGrid->setWindDirX(_x);
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::updateWindDirY(double _y){
    m_oceanGrid->setWindDirY(_y);
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::resetSim(){
    m_oceanGrid->resetSim();
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::updateResolution(QString _res){
    if(_res == QString("128x128")){
        m_oceanGrid->setResolution(128);
    }
    else if (_res == QString("256x256")){
        m_oceanGrid->setResolution(256);
    }
    else if (_res == QString("512x512")){
        m_oceanGrid->setResolution(512);
    }
    else if (_res == QString("1024x1024")){
        m_oceanGrid->setResolution(1024);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::updateAmplitude(double _value){
    m_oceanGrid->setAmplitude(_value);
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::updateBaseColour(QColor _colour){
    m_oceanGrid->setSeaBaseCol(_colour);
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::updateTopColour(QColor _colour){
    m_oceanGrid->setSeaTopCol(_colour);
}
//----------------------------------------------------------------------------------------------------------------------
// Bad way to do this should be improved!!
float3 OpenGLWidget::getSeaTopColour(){
    return m_oceanGrid->getSeaTopColour();
}
//----------------------------------------------------------------------------------------------------------------------
float3 OpenGLWidget::getSeaBaseColour(){
    return m_oceanGrid->getSeaBaseColour();
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::boatCheckBox(){
    if (m_renderBoat){
        m_renderBoat = false;
    }
    else{
        m_renderBoat = true;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::skyboxCheckBox(){
    if (m_renderSkyBox){
        m_renderSkyBox = false;
    }
    else{
        m_renderSkyBox = true;
    }
}
//----------------------------------------------------------------------------------------------------------------------
