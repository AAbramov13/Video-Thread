#ifndef SYNCHROTHREAD_H
#define SYNCHROTHREAD_H

#include <QObject>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

class SynchroThread : public QObject
{
    Q_OBJECT

    Q_PROPERTY(bool running READ running WRITE setRunning NOTIFY runningChanged)
    bool m_running;
    Mat m_image;
    Mat m_image_2;
    double max_dist;
    double min_dist;
    int maxCountBufferImg;
    Mat bufferImg1[];
    Mat bufferImg2[];
    int countBufferImage;
    int countDotArray[][3];
    //[][0-кол-во векторов;1-id img1;2-id img2]

public:
    SynchroThread();
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
    void createCompairArray(Mat bufferImg1[],Mat bufferImg2[]);
    void compairCountControlPoints(int array[][3]);
    int findDescriptors(Mat left_image, Mat right_image);
    void showPoint2fVectors(vector<Point2f> v, string name);
    void showPoint2IntVectors(vector<Point2i> v, string name);
    void showKeyPointVectors(vector<KeyPoint> vLeft, vector<KeyPoint> vRight);
    //vector<Point2i> changeBasicZero (vector<Point2f> v, int size);
    vector<Point2f> changeBasicZero (vector<Point2f> v, int size);
    vector<Point2f> changeVectorFloatToBinary(vector<Point2f> v);
    vector<Point2i> changeVectorIntToBinary(vector<Point2i> v);
    float changeFloatToBinary(float f);
    int changeIntToBinary(int intCount);
    void setRunning(bool running);
    void setImage(Mat image);
    void setImage_2(Mat image2);
    void useMatHomogeneus(vector<KeyPoint> keypoints_object,vector<KeyPoint> keypoints_scene,vector< DMatch > good_matches,Mat img_matches,Mat img_object);
};

#endif // SYNCHROTHREAD_H
