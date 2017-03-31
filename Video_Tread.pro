#-------------------------------------------------
#
# Project created by QtCreator 2017-01-26T14:06:48
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Video_Tread
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    stichingthread.cpp \
    threadobject.cpp

HEADERS  += mainwindow.h \
    stichingthread.h \
    threadobject.h

FORMS    += mainwindow.ui



win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../opencv_build/lib/release/ -lopencv_stitching
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../opencv_build/lib/debug/ -lopencv_stitching

INCLUDEPATH += "/usr/local/include/"
LIBS += -L"/usr/local/lib/"
LIBS += -lopencv_core \
        -lopencv_features2d \
        -lopencv_highgui \
        -lopencv_imgcodecs \
        -lopencv_imgproc \
        -lopencv_video \
        -lopencv_videoio \
        -lopencv_videostab \
        -lopencv_xfeatures2d \
        -lopencv_flann \
        -lopencv_calib3d

