#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <openglwidget.h>
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    OpenGLWidget *m_openGLWidget;

public slots:
    void choppinessValue(int _value);
    void simulationTime(float _time);
    void FPS(float _FPS);


};

#endif // MAINWINDOW_H
