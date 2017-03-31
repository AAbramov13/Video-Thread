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

//initial
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


double StichingThread::getMax_dist() const
{
    return max_dist;
}

void StichingThread::setMax_dist(double value)
{
    max_dist = value;
}

double StichingThread::getMin_dist() const
{
    return min_dist;
}

void StichingThread::setMin_dist(double value)
{
    min_dist = value;
}

StichingThread::StichingThread()
{

}

bool StichingThread::running() const
{
    return m_running;
}

Mat StichingThread::image() const
{
    return m_image;
}

Mat StichingThread::image_2() const
{
    return m_image_2;
}
void StichingThread::run()
{
    while(m_running){
            getRoiRegion(m_image,m_image_2);
    }
//        destroyWindow("result");
}

void showFloatVector(vector<Point2f> v);

void StichingThread::getRoiRegion(Mat img_1, Mat img_2)
{
    //всякая хрень с размерностью
    int height_left_image = img_1.rows;
    int width_left_image = img_1.cols;

    int height_right_image = img_2.rows;
//    int width_right_image = img_2.cols;

    int width_interested_image = width_left_image/3;
    int heighImage = minCount(height_left_image,height_right_image);

    int x_left_image = width_left_image - width_interested_image;
    int y_left_image = 0;

    int x_right_image = 0;
    int y_right_image = 0;

    //--------
    //вырезаем интересующие нас регионы
    Rect interested_region_left_image = Rect(x_left_image,y_left_image,width_interested_image,heighImage);
    Rect interested_region_right_image = Rect(x_right_image,y_right_image,width_interested_image,heighImage);

    Mat img_object = img_1(interested_region_left_image);
    Mat img_scene = img_2(interested_region_right_image);

    findDescriptors(img_object,img_scene);
}

int StichingThread::minCount(int x, int y)
{
    if(x == y) return x;
    else if(x > y) return y;
    else if (x < y) return x;
}

void StichingThread::findDescriptors(Mat img_object, Mat img_scene)
{
    int minHessian = 400;

     Ptr<SURF> detector = SURF::create( minHessian );

    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;

    detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );

    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

    setMax_dist(0);
    setMin_dist(100);

    for( int i = 0; i < descriptors_object.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < getMin_dist() ) setMin_dist(dist);
        if( dist > getMax_dist() ) setMax_dist(dist);
    }

//    printf("-- Max dist : %f \n", max_dist );
//    printf("-- Min dist : %f \n", min_dist );

    vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_object.rows; i++ )
    {
        if( matches[i].distance < 2 * getMin_dist() )
        {
            good_matches.push_back( matches[i]);
        }
    }


    Mat img_matches;

    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
    good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    vector <Point2f> good_key2points_object,good_key2points_scene;
    vector <Point2i> good_keyInt_object, good_keyInt_scene;

    for(int i = 0; i < good_matches.size(); i++){
        good_key2points_object.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        good_key2points_scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }


    good_key2points_object = changeBasicZero(good_key2points_object,good_matches.size());
    good_key2points_scene = changeBasicZero(good_key2points_scene,good_matches.size());

    showPoint2fVectors(good_key2points_object,"obj");

//    good_key2points_object = changeVectorFloatToBinary(good_key2points_object);
//    good_key2points_scene = changeVectorFloatToBinary(good_key2points_scene);

//    showFloatVector(good_key2points_scene);

    useMatHomogeneus(keypoints_object, keypoints_scene, good_matches, img_matches, img_object);
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
        v[i].x = changeFloatToBinary(v[i].x);
        v[i].y = changeFloatToBinary(v[i].y);
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
        binaryTotal = binaryInt +binaryFract;

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

        cout << obj.size() << " "<< scene.size() << endl;
    Mat H = findHomography( obj, scene, CV_RANSAC );

    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows );
    obj_corners[3] = cvPoint( 0, img_object.rows );


    std::vector<Point2f> scene_corners(4);
    //отбражаем углы целевого объекта, используя найденное преобразование, на сцену
    perspectiveTransform( obj_corners, scene_corners, H);

    //соединение отображенных углов
//    line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
//    line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

    imshow( "result", img_matches );
}


