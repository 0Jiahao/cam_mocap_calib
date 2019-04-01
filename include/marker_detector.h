#ifndef MARKER_DETECTOR_H
#define MARKER_DETECTOR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class marker_detector
{
    public:
    //members
    Mat img;
    Point2f o;
    Point2f x;
    Point2f y;
    Point2f f;
    bool isDetected;
    marker_detector();
    void read(Mat img);
    void process();
};

#endif