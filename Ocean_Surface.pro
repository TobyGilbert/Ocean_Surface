cache()
TARGET=Ocean
OBJECTS_DIR=obj

# as I want to support 4.8 and 5 this will set a flag for some of the mac stuff
# mainly in the types.h file for the setMacVisual which is native in Qt5
isEqual(QT_MAJOR_VERSION, 5) {
        cache()
        DEFINES +=QT5BUILD
}
TEMPLATE = app
UI_HEADERS_DIR=ui
MOC_DIR=moc

CONFIG += console
CONFIG-=app_bundle
QT+=gui opengl core
SOURCES += \
    src/main.cpp \
    src/mainwindow.cpp \
    src/Camera.cpp \
    src/ShaderUtils.cpp \
    src/TextureUtils.cpp \
    src/ShaderProgram.cpp \
    src/Texture.cpp \
    src/ModelLoader.cpp \
    src/OpenGLWidget.cpp \
    src/Shader.cpp \
    src/CudaUtils.cpp \
    src/Ocean.cu \
    src/OceanGrid.cpp \
    src/Skybox.cpp \
    src/Sun.cpp

CUDA_SOURCES += src/Ocean.cu
SOURCES -= src/Ocean.cu


HEADERS += \
    include/mainwindow.h \
    include/Camera.h \
    include/ShaderUtils.h \
    include/TextureUtils.h \
    include/ShaderProgram.h \
    include/Texture.h \
    include/ModelLoader.h \
    include/OpenGLWidget.h \
    include/Shader.h \
    include/CudaUtils.h \
    include/Ocean.h \
    include/OceanGrid.h \
    include/Skybox.h \
    include/Sun.h

# use this to suppress some warning from boost
QMAKE_CXXFLAGS_WARN_ON += "-Wno-unused-parameter"
QMAKE_CXXFLAGS+= -msse -msse2 -msse3
macx:QMAKE_CXXFLAGS+= -arch x86_64
macx:INCLUDEPATH+=/usr/local/include/
# define the _DEBUG flag for the graphics lib

unix:LIBS += -L/usr/local/lib

# now if we are under unix and not on a Mac (i.e. linux) define GLEW
linux-*{
                linux-*:QMAKE_CXXFLAGS +=  -march=native
                linux-*:DEFINES+=GL42
                DEFINES += LINUX
}
DEPENDPATH+=include

PROJECT_DIR = $$system(pwd)

# paths to cuda directory
macx:CUDA_DIR = /Developer/NVIDIA/CUDA-6.5

NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

INCLUDEPATH += /usr/local/include
INCLUDEPATH +=./include /opt/local/include
INCLUDEPATH += $$CUDA_DIR/include
INCLUDEPATH += $$CUDA_DIR/samples/common/inc
INCLUDEPATH += $$CUDA_DIR/../shared/inc
# lib dirs
linux:QMAKE_LIBDIR += $$CUDA_DIR/lib64
QMAKE_LIBDIR += $$CUDA_DIR/lib
QMAKE_LIBDIR += $$CUDA_DIR/samples/common/lib
QMAKE_LIBDIR += /opt/local/lib

LIBS += -lassimp -lnoise -lcudart -lcufftw -lcufft

DESTDIR=./

CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# Prepare the extra compiler configuration
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -gencode arch=compute_30,code=sm_30 -c $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}

QMAKE_EXTRA_UNIX_COMPILERS += cuda

# if we are on a mac define DARWIN
macx:DEFINES += DARWIN


FORMS += \
    ui/mainwindow.ui

OTHER_FILES += \
    shaders/PhongFrag.glsl \
    shaders/PhongVert.glsl \
    shaders/CubeMapVert.glsl \
    shaders/CubeMapFrag.glsl \
    shaders/OceanFrag.glsl \
    shaders/OceanVert.glsl \
    textures/skyCubeMap_negx.png \
    textures/skyCubeMap_negy.png \
    textures/skyCubeMap_negz.png \
    textures/skyCubeMap_posx.png \
    textures/skyCubeMap_posy.png \
    textures/skyCubeMap_posz.png \
    textures/interstellar_east.jpg \
    textures/interstellar_north.jpg \
    textures/interstellar_south.jpg \
    textures/interstellar_up.jpg \
    textures/interstellar_west.jpg \
    textures/miramar_negx.jpg \
    textures/miramar_negy.jpg \
    textures/miramar_negz.jpg \
    textures/miramar_posx.jpg \
    textures/miramar_posy.jpg \
    textures/miramar_posz.jpg \
    models/sphere.obj \
    models/cube.obj

