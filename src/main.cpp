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

//    fillGPUArray(gpuFloatArray, 128);

//    float floats[128];
//    cudaMemcpy(floats, gpuFloatArray, 128*sizeof(float), cudaMemcpyDeviceToHost);

//    for (int i=0; i<128; i++){
//        std::cout<<floats[i]<<std::endl;
//    }

    QApplication app(argc,argv);
    MainWindow w;
    w.setWindowTitle(QString("Path Tracer"));
    w.show();
    app.exec();
}
