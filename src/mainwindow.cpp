#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QColorDialog>
#include <QColor>
//-------------------------------------------------------------------------------------------------------------------------
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

    // Changes the resoulution of the grid
    ui->comboBox->addItem(tr("128x128"));
    ui->comboBox->addItem(tr("256x256"));
    ui->comboBox->addItem(tr("512x512"));
    ui->comboBox->addItem(tr("1024x1024"));
    ui->comboBox->setCurrentIndex(0);

    // Chopiness UI
    this->setMinimumWidth(1000);
    ui->choppinessSlider->setMinimum(0);
    ui->choppinessSlider->setSliderPosition(5);
    ui->choppinessSlider->setMaximum(100);
    ui->lineEdit->setText("0.05");

    // FPS and time output line edits
    ui->timeLineEdit->setReadOnly(true);

    // Wind direction x and z
    ui->m_xWindSpinBox->setValue(0.0);
    ui->m_xWindSpinBox->setMaximum(1.0);
    ui->m_zWindSpinBox->setValue(1.0);
    ui->m_zWindSpinBox->setMaximum(1.0);

    // Amplitude
    ui->m_amplitudeSpinBox->setValue((double)5.0);
    ui->m_amplitudeSpinBox->setMaximum(500.0);

    // Checkbox for rendering the skybox
    ui->m_skyboxCheckBox->setChecked(true);

    // Colours for shading
    ui->m_topColourBtn->setFlat(true);
    QPalette palette = ui->m_topColourBtn->palette();
    palette.setColor( QPalette::Button, QColor( 204, 230, 153 ) );
    ui->m_topColourBtn->setPalette(palette);
    ui->m_topColourBtn->setAutoFillBackground(true);

    ui->m_baseColourBtn->setFlat(true);
    palette = ui->m_baseColourBtn->palette();
    palette.setColor( QPalette::Button, QColor( 25, 49 , 56 )  );
    ui->m_baseColourBtn->setPalette(palette);
    ui->m_baseColourBtn->setAutoFillBackground(true);

    connect(ui->m_xWindSpinBox, SIGNAL(valueChanged(double)), m_openGLWidget, SLOT(updateWindDirX(double)));
    connect(ui->m_xWindSpinBox, SIGNAL(editingFinished()), m_openGLWidget, SLOT(resetSim()));
    connect(ui->m_zWindSpinBox, SIGNAL(valueChanged(double)), m_openGLWidget, SLOT(updateWindDirY(double)));
    connect(ui->m_zWindSpinBox, SIGNAL(editingFinished()), m_openGLWidget, SLOT(resetSim()));
    connect(ui->choppinessSlider, SIGNAL(valueChanged(int)), m_openGLWidget, SLOT(updateChoppiness(int)));
    connect(ui->choppinessSlider, SIGNAL(valueChanged(int)), m_openGLWidget, SLOT(resetSim()));
    connect(ui->choppinessSlider, SIGNAL(valueChanged(int)), this, SLOT(choppinessValue(int)));
    connect(m_openGLWidget, SIGNAL(updateTimer(float)), this, SLOT(simulationTime(float)));
    connect(ui->comboBox, SIGNAL(currentIndexChanged(QString)), m_openGLWidget, SLOT(updateResolution(QString)));
    connect(ui->m_amplitudeSpinBox, SIGNAL(valueChanged(double)), m_openGLWidget, SLOT(updateAmplitude(double)));
    connect(ui->m_topColourBtn, SIGNAL(clicked()), this, SLOT(changeSeaTopColourBtn()));
    connect(ui->m_baseColourBtn, SIGNAL(clicked()), this, SLOT(changeSeaBaseColourBtn()));
    connect(ui->m_boatCheckBox, SIGNAL(clicked()), m_openGLWidget, SLOT(boatCheckBox()));
    connect(ui->m_skyboxCheckBox, SIGNAL(clicked()), m_openGLWidget, SLOT(skyboxCheckBox()));

}
//-------------------------------------------------------------------------------------------------------------------------
MainWindow::~MainWindow(){
    delete ui;
    delete m_openGLWidget;
}
//-------------------------------------------------------------------------------------------------------------------------
void MainWindow::choppinessValue(int _value){
    ui->lineEdit->setText(QString::number((float)_value/100.0));
}
//-------------------------------------------------------------------------------------------------------------------------
void MainWindow::simulationTime(float _time){
    ui->timeLineEdit->setText(QString::number(_time));
}
//-------------------------------------------------------------------------------------------------------------------------
void MainWindow::changeSeaBaseColourBtn(){
    QColor col;
    col = QColorDialog::getColor();
    QPalette palette = ui->m_baseColourBtn->palette();
    palette.setColor( QPalette::Button, col  );
    ui->m_baseColourBtn->setPalette(palette);
    ui->m_baseColourBtn->setAutoFillBackground(true);
    m_openGLWidget->updateBaseColour(col);
}
//-------------------------------------------------------------------------------------------------------------------------
void MainWindow::changeSeaTopColourBtn(){
    QColor col;
    col = QColorDialog::getColor();
    QPalette palette = ui->m_topColourBtn->palette();
    palette.setColor( QPalette::Button, col  );
    ui->m_topColourBtn->setPalette(palette);
    ui->m_topColourBtn->setAutoFillBackground(true);
    m_openGLWidget->updateTopColour(col);
}
//-------------------------------------------------------------------------------------------------------------------------
