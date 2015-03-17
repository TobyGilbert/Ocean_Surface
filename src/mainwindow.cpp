#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow){
    ui->setupUi(this);

    QGLFormat format;
    format.setVersion(4,1);
    format.setProfile(QGLFormat::CoreProfile);

    m_openGLWidget = new OpenGLWidget(format,this);
    ui->gridLayout->addWidget(m_openGLWidget,0,0,1,1);

    ui->groupBox->setMaximumWidth(350);

    ui->choppinessSlider->setMinimum(0);
    ui->choppinessSlider->setSliderPosition(2);
    ui->choppinessSlider->setMaximum(10);

    ui->lineEdit->setText("0.02");

    ui->timeLineEdit->setReadOnly(true);

    ui->FPSLineEdit->setReadOnly(true);

    connect(ui->choppinessSlider, SIGNAL(sliderMoved(int)), m_openGLWidget, SLOT(updateChoppiness(int)));
    connect(ui->choppinessSlider, SIGNAL(sliderMoved(int)), this, SLOT(choppinessValue(int)));
    connect(m_openGLWidget, SIGNAL(updateTimer(float)), this, SLOT(simulationTime(float)));
    connect(m_openGLWidget, SIGNAL(updateFPS(float)), this, SLOT(FPS(float)));

}

MainWindow::~MainWindow(){
    delete ui;
    delete m_openGLWidget;
}
void MainWindow::choppinessValue(int _value){
    ui->lineEdit->setText(QString::number((float)_value/100.0));
}

void MainWindow::simulationTime(float _time){
    ui->timeLineEdit->setText(QString::number(_time));
}

void MainWindow::FPS(float _FPS){
    ui->FPSLineEdit->setText(QString::number(_FPS));
}
