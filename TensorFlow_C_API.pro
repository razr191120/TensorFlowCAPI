QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp \
        ctensorflow.cpp

HEADERS += \
    mainwindow.h \
        ctensorflow.h \
	scope_guard.hpp

FORMS += \
    mainwindow.ui

#----------TensorFlow-----------------------#
INCLUDEPATH += ../../CTensorflow/include
LIBS += -L../../CTensorflow/lib/ -ltensorflow -ltensorflow_framework
LIBS += -L/usr/local/cuda-10.0/lib64

#----------OpenCV-------------------------#
INCLUDEPATH += ../../opencv/include
LIBS += -L/../../opencv/lib
LIBS += -lopencv_core -lopencv_imgcodecs -lopencv_highgui \
-lopencv_videoio -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_imgproc -lopencv_dnn
#-------------------------------------------#

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
