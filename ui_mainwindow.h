/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.3.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QGridLayout *gridLayout;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_2;
    QTabWidget *tabWidget;
    QWidget *tab;
    QGridLayout *gridLayout_3;
    QLineEdit *lineEdit;
    QLineEdit *timeLineEdit;
    QLabel *label;
    QLabel *label_5;
    QLabel *label_2;
    QLabel *label_3;
    QLabel *label_4;
    QDoubleSpinBox *doubleSpinBox;
    QSpacerItem *verticalSpacer;
    QLineEdit *FPSLineEdit;
    QDoubleSpinBox *doubleSpinBox_2;
    QLabel *label_6;
    QSlider *choppinessSlider;
    QLabel *label_7;
    QLabel *label_8;
    QComboBox *comboBox;
    QDoubleSpinBox *doubleSpinBox_3;
    QWidget *tab_2;
    QSpacerItem *horizontalSpacer;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(800, 600);
        MainWindow->setMinimumSize(QSize(500, 500));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        gridLayout = new QGridLayout(centralwidget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        groupBox = new QGroupBox(centralwidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        gridLayout_2 = new QGridLayout(groupBox);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        tabWidget = new QTabWidget(groupBox);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        gridLayout_3 = new QGridLayout(tab);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        lineEdit = new QLineEdit(tab);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));

        gridLayout_3->addWidget(lineEdit, 6, 3, 1, 1);

        timeLineEdit = new QLineEdit(tab);
        timeLineEdit->setObjectName(QStringLiteral("timeLineEdit"));

        gridLayout_3->addWidget(timeLineEdit, 0, 1, 1, 1);

        label = new QLabel(tab);
        label->setObjectName(QStringLiteral("label"));

        gridLayout_3->addWidget(label, 0, 0, 1, 1);

        label_5 = new QLabel(tab);
        label_5->setObjectName(QStringLiteral("label_5"));

        gridLayout_3->addWidget(label_5, 8, 2, 1, 1);

        label_2 = new QLabel(tab);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout_3->addWidget(label_2, 4, 0, 1, 1);

        label_3 = new QLabel(tab);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout_3->addWidget(label_3, 0, 2, 1, 1);

        label_4 = new QLabel(tab);
        label_4->setObjectName(QStringLiteral("label_4"));

        gridLayout_3->addWidget(label_4, 8, 0, 1, 1);

        doubleSpinBox = new QDoubleSpinBox(tab);
        doubleSpinBox->setObjectName(QStringLiteral("doubleSpinBox"));

        gridLayout_3->addWidget(doubleSpinBox, 8, 1, 1, 1);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_3->addItem(verticalSpacer, 11, 0, 1, 1);

        FPSLineEdit = new QLineEdit(tab);
        FPSLineEdit->setObjectName(QStringLiteral("FPSLineEdit"));

        gridLayout_3->addWidget(FPSLineEdit, 0, 3, 1, 1);

        doubleSpinBox_2 = new QDoubleSpinBox(tab);
        doubleSpinBox_2->setObjectName(QStringLiteral("doubleSpinBox_2"));

        gridLayout_3->addWidget(doubleSpinBox_2, 8, 3, 1, 1);

        label_6 = new QLabel(tab);
        label_6->setObjectName(QStringLiteral("label_6"));

        gridLayout_3->addWidget(label_6, 7, 0, 1, 1);

        choppinessSlider = new QSlider(tab);
        choppinessSlider->setObjectName(QStringLiteral("choppinessSlider"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(choppinessSlider->sizePolicy().hasHeightForWidth());
        choppinessSlider->setSizePolicy(sizePolicy);
        choppinessSlider->setOrientation(Qt::Horizontal);

        gridLayout_3->addWidget(choppinessSlider, 6, 0, 1, 2);

        label_7 = new QLabel(tab);
        label_7->setObjectName(QStringLiteral("label_7"));

        gridLayout_3->addWidget(label_7, 2, 0, 1, 1);

        label_8 = new QLabel(tab);
        label_8->setObjectName(QStringLiteral("label_8"));

        gridLayout_3->addWidget(label_8, 3, 0, 1, 1);

        comboBox = new QComboBox(tab);
        comboBox->setObjectName(QStringLiteral("comboBox"));

        gridLayout_3->addWidget(comboBox, 3, 1, 1, 2);

        doubleSpinBox_3 = new QDoubleSpinBox(tab);
        doubleSpinBox_3->setObjectName(QStringLiteral("doubleSpinBox_3"));

        gridLayout_3->addWidget(doubleSpinBox_3, 2, 1, 1, 2);

        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        tabWidget->addTab(tab_2, QString());

        gridLayout_2->addWidget(tabWidget, 0, 0, 1, 1);


        gridLayout->addWidget(groupBox, 0, 1, 1, 1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 0, 0, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 800, 25));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QStringLiteral("statusbar"));
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0));
        groupBox->setTitle(QApplication::translate("MainWindow", "Controls", 0));
        label->setText(QApplication::translate("MainWindow", "Time", 0));
        label_5->setText(QApplication::translate("MainWindow", "y:", 0));
        label_2->setText(QApplication::translate("MainWindow", "Choppiness", 0));
        label_3->setText(QApplication::translate("MainWindow", "FPS", 0));
        label_4->setText(QApplication::translate("MainWindow", "x:", 0));
        label_6->setText(QApplication::translate("MainWindow", "Wind Speed", 0));
        label_7->setText(QApplication::translate("MainWindow", "Amplitude", 0));
        label_8->setText(QApplication::translate("MainWindow", "Grid Size", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("MainWindow", "FFT", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("MainWindow", "Gerstner", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
