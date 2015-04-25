#ifdef DARWIN
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
    #include <GL/gl.h>
#endif
#include "Ocean.h"
#include "mainwindow.h"
#include <QApplication>
#include <CudaUtils.h>
#include <cuda_runtime.h>

//#define FULL_SCREEN

int main(int argc, char **argv){
    CudaUtils::printDevices();

//    float *gpuFloatArray;
//    cudaMalloc(&gpuFloatArray, 128*sizeof(float));

    QApplication app(argc,argv);
    MainWindow w;
    w.setWindowTitle(QString("Ocean Surface"));
#ifdef FULL_SCREEN
        w.showMaximized();
#endif

    w.show();
    app.exec();
}
