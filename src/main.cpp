#include <QApplication>
#include "mainwindow.h"
#include "CudaUtils.h"
#include <cuda_runtime.h>
#include "Ocean.h"

int main(int argc, char **argv)
{
    CudaUtils::printDevices();

    float *gpuFloatArray;
    cudaMalloc(&gpuFloatArray, 128*sizeof(float));

    QApplication app(argc,argv);
    MainWindow w;
    w.setWindowTitle(QString("Ocean Surface"));
    w.show();
    app.exec();
}
