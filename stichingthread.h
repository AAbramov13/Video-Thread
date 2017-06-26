#ifndef STICHINGTHREAD_H
#define STICHINGTHREAD_H

#include <QObject>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

class StichingThread : public QObject
{
    Q_OBJECT

    Q_PROPERTY(bool running READ running WRITE setRunning NOTIFY runningChanged)
    bool m_running;
    Mat m_image;
    Mat m_image_2;
    double max_dist;
    double min_dist;
    int maxCountBufferImg;
    Mat* bufferImg1 = new Mat[20];
    Mat* bufferImg2 = new Mat[20];

    int countBufferImage;
    int countDotArray[][3];

public:
    StichingThread();
    bool running() const;
    Mat image() const;
    Mat image_2() const;
    char binaryPoints1[];
    char binaryPoints2[];

    double getMax_dist() const;
    void setMax_dist(double value);

    double getMin_dist() const;
    void setMin_dist(double value);

signals:
    void finished();
    void runningChanged(bool running);
    void imageChanged(Mat image);
    void image_2Changed(Mat image2);


public slots:
    void run();
    void getRoiRegion(Mat img_1,Mat img_2);
    int minCount(int x, int y);
    void createBuffer(Mat img1, Mat img2);
    void caompairImage(Mat bufferImg1[],Mat bufferImg2[]);
    int countHashAndHemming(Mat img1, Mat img2);
    void createCompairArray(Mat bufferImg1[],Mat bufferImg2[]);
    void compairCountControlPoints(int array[][3]);
    int findDescriptors(Mat left_image, Mat right_image);
    void setRunning(bool running);
};

#endif // STICHINGTHREAD_H
