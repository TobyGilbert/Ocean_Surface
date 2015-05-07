/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QGridLayout *gridLayout;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_2;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout_6;
    QLabel *label_9;
    QLabel *label_10;
    QPushButton *m_topColourBtn;
    QPushButton *m_baseColourBtn;
    QLabel *label_12;
    QDoubleSpinBox *doubleSpinBox_4;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_5;
    QLabel *label;
    QLineEdit *timeLineEdit;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout_4;
    QSlider *choppinessSlider;
    QDoubleSpinBox *m_xWindSpinBox;
    QLineEdit *lineEdit;
    QDoubleSpinBox *m_zWindSpinBox;
    QLabel *label_8;
    QLabel *label_7;
    QComboBox *comboBox;
    QLabel *label_2;
    QLabel *label_5;
    QLabel *label_6;
    QDoubleSpinBox *m_amplitudeSpinBox;
    QLabel *label_4;
    QGroupBox *groupBox_5;
    QGridLayout *gridLayout_3;
    QCheckBox *m_boatCheckBox;
    QCheckBox *m_skyboxCheckBox;
    QSpacerItem *horizontalSpacer;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(893, 751);
        MainWindow->setMinimumSize(QSize(500, 500));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        gridLayout = new QGridLayout(centralwidget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        groupBox = new QGroupBox(centralwidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        gridLayout_2 = new QGridLayout(groupBox);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        groupBox_4 = new QGroupBox(groupBox);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        gridLayout_6 = new QGridLayout(groupBox_4);
        gridLayout_6->setObjectName(QStringLiteral("gridLayout_6"));
        label_9 = new QLabel(groupBox_4);
        label_9->setObjectName(QStringLiteral("label_9"));

        gridLayout_6->addWidget(label_9, 0, 0, 1, 1);

        label_10 = new QLabel(groupBox_4);
        label_10->setObjectName(QStringLiteral("label_10"));

        gridLayout_6->addWidget(label_10, 2, 0, 1, 1);

        m_topColourBtn = new QPushButton(groupBox_4);
        m_topColourBtn->setObjectName(QStringLiteral("m_topColourBtn"));

        gridLayout_6->addWidget(m_topColourBtn, 0, 1, 1, 1);

        m_baseColourBtn = new QPushButton(groupBox_4);
        m_baseColourBtn->setObjectName(QStringLiteral("m_baseColourBtn"));

        gridLayout_6->addWidget(m_baseColourBtn, 2, 1, 1, 1);

        label_12 = new QLabel(groupBox_4);
        label_12->setObjectName(QStringLiteral("label_12"));

        gridLayout_6->addWidget(label_12, 3, 0, 1, 1);

        doubleSpinBox_4 = new QDoubleSpinBox(groupBox_4);
        doubleSpinBox_4->setObjectName(QStringLiteral("doubleSpinBox_4"));

        gridLayout_6->addWidget(doubleSpinBox_4, 3, 1, 1, 1);


        gridLayout_2->addWidget(groupBox_4, 2, 0, 1, 1);

        groupBox_3 = new QGroupBox(groupBox);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        gridLayout_5 = new QGridLayout(groupBox_3);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        label = new QLabel(groupBox_3);
        label->setObjectName(QStringLiteral("label"));

        gridLayout_5->addWidget(label, 0, 0, 1, 1);

        timeLineEdit = new QLineEdit(groupBox_3);
        timeLineEdit->setObjectName(QStringLiteral("timeLineEdit"));

        gridLayout_5->addWidget(timeLineEdit, 0, 1, 1, 1);


        gridLayout_2->addWidget(groupBox_3, 0, 0, 1, 1);

        groupBox_2 = new QGroupBox(groupBox);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        gridLayout_4 = new QGridLayout(groupBox_2);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        choppinessSlider = new QSlider(groupBox_2);
        choppinessSlider->setObjectName(QStringLiteral("choppinessSlider"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(choppinessSlider->sizePolicy().hasHeightForWidth());
        choppinessSlider->setSizePolicy(sizePolicy);
        choppinessSlider->setOrientation(Qt::Horizontal);

        gridLayout_4->addWidget(choppinessSlider, 3, 0, 1, 1);

        m_xWindSpinBox = new QDoubleSpinBox(groupBox_2);
        m_xWindSpinBox->setObjectName(QStringLiteral("m_xWindSpinBox"));

        gridLayout_4->addWidget(m_xWindSpinBox, 5, 1, 1, 1);

        lineEdit = new QLineEdit(groupBox_2);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));

        gridLayout_4->addWidget(lineEdit, 3, 1, 1, 1);

        m_zWindSpinBox = new QDoubleSpinBox(groupBox_2);
        m_zWindSpinBox->setObjectName(QStringLiteral("m_zWindSpinBox"));

        gridLayout_4->addWidget(m_zWindSpinBox, 6, 1, 1, 1);

        label_8 = new QLabel(groupBox_2);
        label_8->setObjectName(QStringLiteral("label_8"));

        gridLayout_4->addWidget(label_8, 1, 0, 1, 1);

        label_7 = new QLabel(groupBox_2);
        label_7->setObjectName(QStringLiteral("label_7"));

        gridLayout_4->addWidget(label_7, 0, 0, 1, 1);

        comboBox = new QComboBox(groupBox_2);
        comboBox->setObjectName(QStringLiteral("comboBox"));

        gridLayout_4->addWidget(comboBox, 1, 1, 1, 1);

        label_2 = new QLabel(groupBox_2);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout_4->addWidget(label_2, 2, 0, 1, 1);

        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QStringLiteral("label_5"));

        gridLayout_4->addWidget(label_5, 6, 0, 1, 1);

        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QStringLiteral("label_6"));

        gridLayout_4->addWidget(label_6, 4, 0, 1, 1);

        m_amplitudeSpinBox = new QDoubleSpinBox(groupBox_2);
        m_amplitudeSpinBox->setObjectName(QStringLiteral("m_amplitudeSpinBox"));

        gridLayout_4->addWidget(m_amplitudeSpinBox, 0, 1, 1, 1);

        label_4 = new QLabel(groupBox_2);
        label_4->setObjectName(QStringLiteral("label_4"));

        gridLayout_4->addWidget(label_4, 5, 0, 1, 1);


        gridLayout_2->addWidget(groupBox_2, 1, 0, 1, 1);

        groupBox_5 = new QGroupBox(groupBox);
        groupBox_5->setObjectName(QStringLiteral("groupBox_5"));
        gridLayout_3 = new QGridLayout(groupBox_5);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        m_boatCheckBox = new QCheckBox(groupBox_5);
        m_boatCheckBox->setObjectName(QStringLiteral("m_boatCheckBox"));

        gridLayout_3->addWidget(m_boatCheckBox, 0, 0, 1, 1);

        m_skyboxCheckBox = new QCheckBox(groupBox_5);
        m_skyboxCheckBox->setObjectName(QStringLiteral("m_skyboxCheckBox"));

        gridLayout_3->addWidget(m_skyboxCheckBox, 1, 0, 1, 1);


        gridLayout_2->addWidget(groupBox_5, 3, 0, 1, 1);


        gridLayout->addWidget(groupBox, 0, 4, 1, 1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 0, 0, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 893, 22));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QStringLiteral("statusbar"));
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0));
        groupBox->setTitle(QApplication::translate("MainWindow", "Controls", 0));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "Shading Options", 0));
        label_9->setText(QApplication::translate("MainWindow", "Ocean Top Colour", 0));
        label_10->setText(QApplication::translate("MainWindow", "Ocean Base Colour", 0));
        m_topColourBtn->setText(QString());
        m_baseColourBtn->setText(QString());
        label_12->setText(QApplication::translate("MainWindow", "Streak Width", 0));
        groupBox_3->setTitle(QString());
        label->setText(QApplication::translate("MainWindow", "Simulation Time", 0));
        groupBox_2->setTitle(QApplication::translate("MainWindow", "Wave Options", 0));
        label_8->setText(QApplication::translate("MainWindow", "Grid Size", 0));
        label_7->setText(QApplication::translate("MainWindow", "Amplitude", 0));
        label_2->setText(QApplication::translate("MainWindow", "Choppiness", 0));
        label_5->setText(QApplication::translate("MainWindow", "z:", 0));
        label_6->setText(QApplication::translate("MainWindow", "Wind Direction", 0));
        label_4->setText(QApplication::translate("MainWindow", "x:", 0));
        groupBox_5->setTitle(QString());
        m_boatCheckBox->setText(QApplication::translate("MainWindow", "Boat Visable", 0));
        m_skyboxCheckBox->setText(QApplication::translate("MainWindow", "Skybox Visable", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
