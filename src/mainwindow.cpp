#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow){
    ui->setupUi(this);

    QGLFormat format;
    format.setVersion(4,1);
    format.setProfile(QGLFormat::CoreProfile);

    // Our opengl widget for drawing
    m_openGLWidget = new OpenGLWidget(format,this);
    m_openGLWidget->setMinimumWidth(600);
    ui->gridLayout->addWidget(m_openGLWidget,0,0,1,1);

    ui->groupBox->setMaximumWidth(500);

    // Grid size combobox
    ui->comboBox->addItem(tr("128x128"));
    ui->comboBox->addItem(tr("256x256"));
    ui->comboBox->addItem(tr("512x512"));
    ui->comboBox->addItem(tr("1024x1024"));

    ui->comboBox->setCurrentIndex(1);

    // Chopiness UI
    this->setMinimumWidth(1000);
    ui->choppinessSlider->setMinimum(0);
    ui->choppinessSlider->setSliderPosition(5);
    ui->choppinessSlider->setMaximum(100);
    ui->lineEdit->setText("0.05");

    ui->timeLineEdit->setReadOnly(true);

    ui->FPSLineEdit->setReadOnly(true);

    // Wind direction x and z
    ui->doubleSpinBox->setValue(0.0);
    ui->doubleSpinBox_2->setValue(1.0);


    connect(ui->doubleSpinBox, SIGNAL(valueChanged(double)), m_openGLWidget, SLOT(updateWindDirX(double)));
    connect(ui->doubleSpinBox, SIGNAL(editingFinished()), m_openGLWidget, SLOT(resetSim()));
    connect(ui->doubleSpinBox_2, SIGNAL(valueChanged(double)), m_openGLWidget, SLOT(updateWindDirY(double)));
    connect(ui->doubleSpinBox_2, SIGNAL(editingFinished()), m_openGLWidget, SLOT(resetSim()));
    connect(ui->choppinessSlider, SIGNAL(valueChanged(int)), m_openGLWidget, SLOT(updateChoppiness(int)));
    connect(ui->choppinessSlider, SIGNAL(valueChanged(int)), m_openGLWidget, SLOT(resetSim()));
    connect(ui->choppinessSlider, SIGNAL(valueChanged(int)), this, SLOT(choppinessValue(int)));
    connect(m_openGLWidget, SIGNAL(updateTimer(float)), this, SLOT(simulationTime(float)));
    connect(m_openGLWidget, SIGNAL(updateFPS(float)), this, SLOT(FPS(float)));
    connect(ui->comboBox, SIGNAL(currentIndexChanged(QString)), m_openGLWidget, SLOT(changeResolution(QString)));

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
