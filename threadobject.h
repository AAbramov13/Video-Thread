#ifndef THREADOBJECT_H
#define THREADOBJECT_H

#include <QObject>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

class ThreadObject : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool running READ running WRITE setRunning NOTIFY runningChanged)
    Q_PROPERTY(QString message READ message WRITE setMessage NOTIFY messageChanged)
    Q_PROPERTY(QString frame_name READ frame_name WRITE setFrame_name NOTIFY frame_nameChanged)


    bool m_running;
    QString m_message;
    QString m_frame_name;

public:
    explicit ThreadObject(QObject *parent = 0);
    bool running() const;
    QString message() const;
    QString frame_name() const;

signals:
    void finished();
    void runningChanged(bool running);
    void messageChanged(QString message);
    void frame_nameChanged(QString frame_name);
    void sendImage(Mat image);

public slots:
    void run();
    void setRunning(bool running);
    void setMessage(QString message);
    void setFrame_name(QString frame_name);
};

#endif // THREADOBJECT_H
