#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "marker_detector.h"

using namespace std;
using namespace cv;

marker_detector::marker_detector()
{
    this->o = Point2f(0,0);
    this->x = Point2f(0,0);
    this->y = Point2f(0,0);
    this->f = Point2f(0,0);
}

void marker_detector::read(Mat img)
{
    // convert RGB to GRAY
    cvtColor(img, this->img, CV_RGB2GRAY);

    // convert GRAY to BW
    // adaptiveThreshold(this->img, this->img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 10);
    threshold(this->img, this->img, 127, 255, THRESH_BINARY);
}

void marker_detector::process()
{
    this->isDetected = false;
    // find contours in the image
    vector<vector<Point> > contours;
    vector<Point2f> centers;
    vector<Vec4i> hierarchy;
    findContours(this->img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    for(int i = 0; i < hierarchy.size(); i++)
    {
		Point2f temp = Point2f(0,0);
		for(int j = 0; j < contours[i].size(); j++)
        {
            temp.x = temp.x + contours[i][j].x / float(contours[i].size());
            temp.y = temp.y + contours[i][j].y / float(contours[i].size());
        }
        centers.push_back(temp);
	}
	cvtColor(this->img, this->img, CV_GRAY2RGB);
    vector<int> lvl;
    for(int i = 0; i < hierarchy.size(); i++)
    {
        int son_idx = hierarchy[i][3];
        int num_lvl = 0;
        while(son_idx != -1)
        {
            son_idx = hierarchy[son_idx][3];
            num_lvl++;
        }
        lvl.push_back(num_lvl);
    }
    // find x end
    int x_end_idx = -1;
    int max_lvl = 0;
    for(int i = 0; i < hierarchy.size(); i++)
    {
        if(hierarchy[i][2] == -1)
        {
            if(lvl[i] > max_lvl)
            {
                max_lvl = lvl[i];
                x_end_idx = i;
            }
        }
    }
    if(x_end_idx >=0)
    {
		drawContours( this->img, contours, x_end_idx, Scalar(0,0,255), 2, 8, hierarchy, 0, Point() );
		Point2f x_end = centers[x_end_idx];
        // find org
        int org_idx = -1;
		double min_dist_to_x_end = 800;
		for(int i = 0; i < hierarchy.size(); i++)
		{
			if(lvl[i] >= max_lvl - 1 && i != x_end_idx && hierarchy[i][2] != x_end_idx && norm(centers[i] - x_end) < min_dist_to_x_end)
			{
				org_idx = i;
				min_dist_to_x_end = norm(centers[i] - x_end);
			}
		}
		if(org_idx >= 0)
		{
			drawContours( this->img, contours, org_idx, Scalar(0,255,0), 2, 8, hierarchy, 0, Point() );
			Point2f org = centers[org_idx];
			// find main marker
			int main_idx = -1;
			main_idx = hierarchy[hierarchy[org_idx][3]][3];			
			if(main_idx >= 0)
			{
				drawContours( this->img, contours, hierarchy[hierarchy[org_idx][3]][3], Scalar(255,255,0), 2, 8, hierarchy, 0, Point() );
				// extract corners
				int max_dist_to_x_end = 0;
				int max_dist_to_org = 0;
				for(int i = 0; i < contours[main_idx].size(); i++)
				{
					double dist_to_x_end = norm(Point2f(contours[main_idx][i].x,contours[main_idx][i].y) - x_end);
					double dist_to_org = norm(Point2f(contours[main_idx][i].x,contours[main_idx][i].y) - org);
					if(dist_to_x_end > max_dist_to_x_end)
					{
						max_dist_to_x_end = dist_to_x_end;
						this->y = Point2f(contours[main_idx][i].x,contours[main_idx][i].y); 
					}
					if(dist_to_org > max_dist_to_org)
					{
						max_dist_to_org = dist_to_org;
						this->o = Point2f(contours[main_idx][i].x,contours[main_idx][i].y); 
					}
				}
				int max_dist_to_y = 0;
				int max_dist_to_o = 0;
				for(int i = 0; i < contours[main_idx].size(); i++)
				{
					double dist_to_o = norm(Point2f(contours[main_idx][i].x,contours[main_idx][i].y) - this->o);
					double dist_to_y = norm(Point2f(contours[main_idx][i].x,contours[main_idx][i].y) - this->y);
					if(dist_to_y > max_dist_to_y)
					{
						max_dist_to_y = dist_to_y;
						this->x = Point2f(contours[main_idx][i].x,contours[main_idx][i].y);
					}
					if(dist_to_o > max_dist_to_o)
					{
						max_dist_to_o = dist_to_o;
						this->f = Point2f(contours[main_idx][i].x,contours[main_idx][i].y);
					}
				}
				circle(this->img, this->o, 3, Scalar(0,255,0), 5, 8, 0);
				circle(this->img, this->x, 3, Scalar(0,0,255), 5, 8, 0);
				circle(this->img, this->y, 3, Scalar(255,0,0), 5, 8, 0);
				circle(this->img, this->f, 3, Scalar(255,255,0), 5, 8, 0);
				if(norm(this->o - this->x) > 10 && norm(this->o - this->y) > 10 && norm(this->x - this->y) > 10 && norm(this->o - this->f) * norm(this->x - this->y) > 300 && norm(x_end - org) > 0.5 * norm(this->x - this->f))
				{
					this->isDetected = true;
				}
			}
		}
	}
}
