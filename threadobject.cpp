#include "threadobject.h"
#include <QApplication>
#include <QThread>
#include <QMutex>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

ThreadObject::ThreadObject(QObject *parent) :
    QObject(parent),
    m_message(""),
    m_frame_name("")
{

}

bool ThreadObject::running() const
{
    return m_running;
}

QString ThreadObject::message() const
{
    return m_message;
}

QString ThreadObject::frame_name() const
{
    return m_frame_name;
}

void ThreadObject::run()
{
    while(m_running)
    {

        cout << "Start tread."<< endl;

        QByteArray ba = m_message.toLocal8Bit();
        std::string addressCam = std::string(ba.data(), ba.size());

        ba = m_frame_name.toLocal8Bit();
        std::string frame_name = std::string(ba.data(), ba.size());
        
        VideoCapture vcap;
        Mat image;
       
        if(!vcap.open(addressCam)) {
            std::cout << "Error opening video 2 stream or file" << endl;
            return -1;
            break;
        }

        cout << "start send image"<< endl;
        while(m_running){
        if(!vcap.read(image)) {
            cout << "No frame" << endl;
            waitKey();
        }
        emit sendImage(image);
        imshow(frame_name, image);
        }
        if(m_running!=true){
            destroyWindow(frame_name);
        }
    }
    emit finished();
}


//все сеты

void ThreadObject::setRunning(bool running)
{
    if (m_running == running)
        return;

    m_running = running;
    emit runningChanged(running);
}

void ThreadObject::setMessage(QString message)
{
    if (m_message == message)
        return;

    m_message = message;
    emit messageChanged(message);
}

void ThreadObject::setFrame_name(QString frame_name)
{
    if (m_frame_name == frame_name)
        return;

    m_frame_name = frame_name;
    emit frame_nameChanged(frame_name);
}
