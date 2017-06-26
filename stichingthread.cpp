#include "stichingthread.h"
#include <QApplication>
#include <QThread>
#include <QMutex>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/xfeatures2d/cuda.hpp>
#include "opencv2/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

// рассчитать хеш картинки
int64 calcImageHash(IplImage* image, bool show_results=false);
// рассчёт расстояния Хэмминга
int64 calcHammingDistance(int64 x, int64 y);

//initial
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


double SynchroThread::getMax_dist() const
{
    return max_dist;
}

void SynchroThread::setMax_dist(double value)
{
    max_dist = value;
}

double SynchroThread::getMin_dist() const
{
    return min_dist;
}

void SynchroThread::setMin_dist(double value)
{
    min_dist = value;
}

SynchroThread::StichingThread()
{

}

bool SynchroThread::running() const
{
    return m_running;
}

Mat SynchroThread::image() const
{
    return m_image;
}

Mat SynchroThread::image_2() const
{
    return m_image_2;
}
void SynchroThread::run()
{
    cout << "start 2nd tread"<<endl;

    countBufferImage =0;
    maxCountBufferImg=20;
    while(m_running){
            getRoiRegion(m_image,m_image_2);
    }
//        destroyWindow("result");
}

void showFloatVector(vector<Point2f> v);

void SynchroThread::getRoiRegion(Mat img_1, Mat img_2)
{
    //всякая хрень с размерностью
    int height_left_image = img_1.rows;
    int width_left_image = img_1.cols;
    int height_right_image = img_2.rows;
    int width_interested_image = width_left_image/3;
    int heighImage = minCount(height_left_image,height_right_image);
    int x_left_image = width_left_image - width_interested_image;
    int y_left_image = 0;
    int x_right_image = 0;
    int y_right_image = 0;
    Rect interested_region_left_image = Rect(x_left_image,y_left_image,width_interested_image,heighImage);
    Rect interested_region_right_image = Rect(x_right_image,y_right_image,width_interested_image,heighImage);

    Mat img_object = img_1(interested_region_left_image);
    Mat img_scene = img_2(interested_region_right_image);

    createBuffer(img_object,img_scene);
}

int SynchroThread::minCount(int x, int y)
{
    if(x == y) return x;
    else if(x > y) return y;
    else if (x < y) return x;
}

void SynchroThread::createBuffer(Mat img1, Mat img2)
{
    char c;
    if(countBufferImage<maxCountBufferImg){
    img1.copyTo(bufferImg1[countBufferImage],img1);
    img2.copyTo(bufferImg2[countBufferImage],img2);
    countBufferImage++;
    c = cvWaitKey(40);
    cout<< "count buffer in image " << countBufferImage<<endl;
    }else{
        m_running = false;
        caompairImage(bufferImg1,bufferImg2);
    }

}

void SynchroThread::caompairImage(Mat bufferImg1[], Mat bufferImg2[])
{
    char c;
    int k = 0;
    for(int i = 0; i < maxCountBufferImg;i++){
        for (int j = 0; j < maxCountBufferImg;j++){
            c=cvWaitKey(40);
            imshow("buffer",bufferImg1[i]);
            imshow("buffer2",bufferImg2[j]);
            countDotArray[k][0]= countHashAndHemming(bufferImg1[i],bufferImg2[j]);
            countDotArray[k][1]= i;
            countDotArray[k][2] = j;
            cout << countDotArray[k][0] << " "<<countDotArray[k][1]<<" "<<countDotArray[k][2]<<endl;
            k++;
        }
    }
    compairCountControlPoints(countDotArray);
}

int SynchroThread::countHashAndHemming(Mat img1, Mat img2)
{
    IplImage* object = new IplImage(img1);
    IplImage* image = new IplImage(img2);

    // построим хэш
    int64 hashO = calcImageHash(object, true);
    int64 hashI = calcImageHash(image, false);
    // рассчитаем расстояние Хэмминга
    int64 dist = calcHammingDistance(hashO, hashI);

    return dist;
}


// рассчитать хеш картинки
int64 calcImageHash(IplImage* src, bool show_results)
{
        if(!src){
                return 0;
        }

        IplImage *res=0, *gray=0, *bin =0;

        res = cvCreateImage( cvSize(8, 8), src->depth, src->nChannels);
        gray = cvCreateImage( cvSize(8, 8), IPL_DEPTH_8U, 1);
        bin = cvCreateImage( cvSize(8, 8), IPL_DEPTH_8U, 1);
        cvResize(src, res);
        cvCvtColor(res, gray, CV_BGR2GRAY);
        CvScalar average = cvAvg(gray);
        cvThreshold(gray, bin, average.val[0], 255, CV_THRESH_BINARY);
        int64 hash = 0;
        int i=0;
        for( int y=0; y<bin->height; y++ ) {
                uchar* ptr = (uchar*) (bin->imageData + y * bin->widthStep);
                for( int x=0; x<bin->width; x++ ) {
                        // 1 канал
                        if(ptr[x]){
                                 hash |= 1<<i;
                        }
                        i++;
                }
        }
        // освобождаем ресурсы
        cvReleaseImage(&res);
        cvReleaseImage(&gray);
        cvReleaseImage(&bin);

        return hash;
}

int64 calcHammingDistance(int64 x, int64 y)
{
        int64 dist = 0, val = x ^ y;

        // Count the number of set bits
        while(val)
        {
                ++dist;
                val &= val - 1;
        }

        return dist;
}



void SynchroThread::createCompairArray(Mat bufferImg1[], Mat bufferImg2[])
{
    cout << "start compair max count"<<endl;
    char c;
    int k = 0;
    for(int i = 0; i < maxCountBufferImg;i++){
        for (int j = 0; j < maxCountBufferImg;j++){
            c=cvWaitKey(10);
            imshow("buffer",bufferImg1[i]);
            imshow("buffer2",bufferImg2[j]);
            countDotArray[k][0]= findDescriptors(bufferImg1[i],bufferImg2[j]);
            countDotArray[k][1]= i;
            countDotArray[k][2] = j;
            cout << countDotArray[k][0] << " "<<countDotArray[k][1]<<" "<<countDotArray[k][2]<<endl;
            k++;

        }
    }

    cout << "end compair max count"<<endl;
    compairCountControlPoints(countDotArray);
}

void SynchroThread::compairCountControlPoints(int array[][3])
{
    cout << "start compair control points"<<endl;

    int maxCountControlPoints[3]={100,0,0};
    for(int i = 0; i< maxCountBufferImg*maxCountBufferImg;i++){
        if(array[i][0]<=maxCountControlPoints[0]){
            maxCountControlPoints[0] = array[i][0];
            maxCountControlPoints[1] = array[i][1];
            maxCountControlPoints[2] = array[i][2];
        }
    }

    cout<<maxCountControlPoints[0]<<" "<<maxCountControlPoints[1]<<" "<<maxCountControlPoints[2]<<endl;

}

int SynchroThread::findDescriptors(Mat img1, Mat img2)
{
     Mat homography;
    FileStorage fs("../data/H1to3p.xml", FileStorage::READ);
    fs.getFirstTopLevelNode() >> homography;
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);
    vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    vector<DMatch> good_matches;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }
    for(unsigned i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;
        col = homography * col;
        col /= col.at<double>(2);
        double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched2[i].pt.y, 2));
        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }

    return good_matches.size();
}

void showFloatVector(vector<Point2f> v)
{
    printf("{");
    for(int i = 0; i < v.size(); i++){
        printf("%d: %lf; %lf\n",i, v[i].x,v[i].y);
    }
    printf("}\n");
}

void StichingThread::showPoint2fVectors(vector<Point2f> v, string name)
{
        cout << name << endl;
    for(int i = 0; i < v.size(); i++){
        cout << i << v[i] << endl;
    }
    cout << " "<<endl;
}

void StichingThread::showPoint2IntVectors(vector<Point2i> v, string name)
{
    cout << name << endl;
for(int i = 0; i < v.size(); i++){
    cout << i << v[i] << endl;
}
cout << " "<<endl;
}

void StichingThread::showKeyPointVectors(vector<KeyPoint> vLeft, vector<KeyPoint> vRight)
{
    int sizeVector = minCount(vLeft.size(),vRight.size());

    for (int i = 0; i < sizeVector; i++){
          printf("%d {X : %f left, %f right - Y: %f left, %f right} \n", i, vLeft[i].pt.x,vRight[i].pt.x,vLeft[i].pt.y,vRight[i].pt.y );
    }

}

//vector<Point2i> StichingThread::changeBasicZero(vector<Point2f> v, int size)
//{
//    float nullCoordinates[2] = {v[0].x,v[0].y};
//    vector<Point2i> vInt;

//    for(int i = 0; i < size; i++)
//    {
//        vInt.push_back(Point2i( (int)(v[i].x - nullCoordinates[0]),(int)(v[i].y - nullCoordinates[1])));
//    }
//    return vInt;
//}

vector<Point2f> StichingThread::changeBasicZero(vector<Point2f> v, int size)
{

    float nullCoordinates[2] = {v[0].x,v[0].y};

    for(int i = 0; i < size; i++)
    {
        v[i].x = v[i].x - nullCoordinates[0];
        v[i].y = v[i].y - nullCoordinates[1];
    }
    return v;
}

vector<Point2f> StichingThread::changeVectorFloatToBinary(vector<Point2f> v)
{
    for(int i = 0; i < v.size(); i++){
        if(v[i].x<0){
            v[i].x = v[i].x*(-1);
            v[i].x = changeFloatToBinary(v[i].x);
        }else{
            v[i].x = changeFloatToBinary(v[i].x);
        }
        if(v[i].y<0){
            v[i].y = v[i].y*(-1);
            v[i].y = changeFloatToBinary(v[i].y);
        }else{
            v[i].y = changeFloatToBinary(v[i].y);
        }
    }
    return v;
}

vector<Point2i> StichingThread::changeVectorIntToBinary(vector<Point2i> v)
{
    for(int i = 0; i < v.size(); i++){
//        v[i].x = changeIntToBinary(v[i].x);
//        v[i].y = changeIntToBinary(v[i].y);
    }
    return v;
}

float StichingThread::changeFloatToBinary(float f)
{
    int  integral, binaryInt = 0, i = 1;
    float  binaryFract = 0, k =0.1f, fractional, temp1, binaryTotal;

    //Separating the integral value from the floating point variable
        integral = (int)f;

        //Separating the fractional value from the variable
        fractional = f - (int)f;

        //Loop for converting decimal to binary
        while(integral>0)
        {
            binaryInt = binaryInt + integral % 2 * i;
            i = i * 10;
            integral = integral / 2;
        }

        //Loop for converting Fractional value to binary
        while(k>0.00000001)
        {
            temp1 = fractional *2;
            binaryFract = binaryFract+((int)temp1)*k;
            fractional = temp1 - (int)temp1;
            k = k / 10;
        }

        //Combining both the integral and fractional binary value.
        binaryTotal = binaryInt+binaryFract;

        return binaryTotal;
}

int StichingThread::changeIntToBinary(int intCount)
{
    int remainder, digits = 0, dividend = intCount;
    while(dividend != 0)
        {
            dividend = dividend / 2;
            digits++;
        }
    int array[digits];
        dividend = intCount;
        for(int i = digits - 1; i >= 0; i--)
        {
            remainder = dividend % 2;
            array[i] = remainder;
            dividend = dividend / 2;
        }
     return dividend/*array*/;
}



void StichingThread::setRunning(bool running)
{
    if (m_running == running)
        return;

    m_running = running;
}

void StichingThread::setImage(Mat image)
{
    m_image = image;
}

void StichingThread::setImage_2(Mat image2)
{
    m_image_2 = image2;
}

void StichingThread::useMatHomogeneus(vector<KeyPoint> keypoints_object, vector<KeyPoint> keypoints_scene, vector<DMatch> good_matches, Mat img_matches,Mat img_object)
{

    vector<Point2f> obj;
    vector<Point2f> scene;

    for( int i = 0; i < good_matches.size(); i++ )
    {
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

        //cout << obj.size() << " "<< scene.size() << endl;
    Mat H = findHomography( obj, scene, CV_RANSAC );

    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows );
    obj_corners[3] = cvPoint( 0, img_object.rows );


    //std::vector<Point2f> scene_corners(4);
    //отбражаем углы целевого объекта, используя найденное преобразование, на сцену
    //erspectiveTransform( obj_corners, scene_corners, H);

    //соединение отображенных углов
//    line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
//    line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

    imshow( "result", img_matches );
}
