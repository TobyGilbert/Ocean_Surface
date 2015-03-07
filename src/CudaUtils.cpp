#include "CudaUtils.h"
//----------------------------------------------------------------------------------------------------------------------
#include <iostream>
#include <QString>
#include <cuda_runtime.h>
//----------------------------------------------------------------------------------------------------------------------
void CudaUtils::printDevices(){
    int count;
    if(cudaGetDeviceCount(&count)){
        return;
    }
    std::cout<<"Found "<<count<<" CUDA device(s)"<<std::endl;
    for (int i=0; i<count; i++){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        QString deviceString = QString("* %1, Compute capability: %2.%3").arg(prop.name).arg(prop.major).arg(prop.minor);
        QString propString1 = QString("  Global mem: %1M, Shared mem per block: %2k, Registers per block: %3").arg(prop.totalGlobalMem / 1024 / 1024)
                .arg(prop.sharedMemPerBlock / 1024).arg(prop.regsPerBlock);
        QString propString2 = QString("  Warp size: %1 threads, Max threads per block: %2, Multiprocessor count: %3").arg(prop.warpSize)
                .arg(prop.maxThreadsPerBlock).arg(prop.multiProcessorCount);
        std::cout<<deviceString.toStdString()<<std::endl;
        std::cout<<propString1.toStdString()<<std::endl;
        std::cout<<propString2.toStdString()<<std::endl;
    }
}
//----------------------------------------------------------------------------------------------------------------------
