#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>
#include <Eigen/Dense>
#include "marker_detector.h"
#include <tf/transform_broadcaster.h>
#include <iostream>
#include <fstream>

using namespace cv; 
using namespace std;
using namespace Eigen;

void getQuaternion(Mat R, double Q[])
{
    double trace = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);
 
    if (trace > 0.0) 
    {
        double s = sqrt(trace + 1.0);
        Q[3] = (s * 0.5);
        s = 0.5 / s;
        Q[0] = ((R.at<double>(2,1) - R.at<double>(1,2)) * s);
        Q[1] = ((R.at<double>(0,2) - R.at<double>(2,0)) * s);
        Q[2] = ((R.at<double>(1,0) - R.at<double>(0,1)) * s);
    } 
    
    else 
    {
        int i = R.at<double>(0,0) < R.at<double>(1,1) ? (R.at<double>(1,1) < R.at<double>(2,2) ? 2 : 1) : (R.at<double>(0,0) < R.at<double>(2,2) ? 2 : 0); 
        int j = (i + 1) % 3;  
        int k = (i + 2) % 3;

        double s = sqrt(R.at<double>(i, i) - R.at<double>(j,j) - R.at<double>(k,k) + 1.0);
        Q[i] = s * 0.5;
        s = 0.5 / s;

        Q[3] = (R.at<double>(k,j) - R.at<double>(j,k)) * s;
        Q[j] = (R.at<double>(j,i) + R.at<double>(i,j)) * s;
        Q[k] = (R.at<double>(k,i) + R.at<double>(i,k)) * s;
    }
}

class my_detector
{  
	public:  
		my_detector()  
		{  
			this->is_calibrated = false;
			image_transport::ImageTransport it(nh);
			//Topic subscribed 
			imgsub = it.subscribe("/camera/color/image_raw", 1, &my_detector::run,this);
			camsub = nh.subscribe("/camera/color/camera_info", 1, &my_detector::store_cam_info,this);
			mocap_marker_o_sub = nh.subscribe("/calib_o/pose", 1, &my_detector::store_mocap_o,this);
			mocap_marker_x_sub = nh.subscribe("/calib_x/pose", 1, &my_detector::store_mocap_x,this);
			mocap_marker_y_sub = nh.subscribe("/calib_y/pose", 1, &my_detector::store_mocap_y,this);
			mocap_marker_cam_sub = nh.subscribe("/calib_cam/pose", 1, &my_detector::store_mocap_cam,this);
			estposepub = nh.advertise<geometry_msgs::PoseStamped>("/cam_mocap_calib/estPose",1);
			bool_sub = nh.subscribe("/cam_mocap_calib/stop", 1, &my_detector::stop,this);
			// meaposepub = nh.advertise<geometry_msgs::PoseStamped>("/cam_mocap_calib/meaPose",1);
			this->image = Mat::zeros(480, 640, CV_8UC3);
			this->mocap_o_detected = false;
			this->mocap_x_detected = false;
			this->mocap_y_detected = false;
			this->mocap_cam_detected = false;
			this->alpha = 0.01;
			this->dR << 1, 0, 0, 0, 1, 0, 0, 0, 1;
			this->dT << 0, 0, 0;			
			// this->myfile.open("my_dataset/calib_data.csv");
			// this->myfile << "meaT.x,meaT.y,meaT.z,meaR.x,meaR.y,meaR.z,meaR.w,estT.x,estT.y,estT.z,estR.x,estR.y,estR.z,estR.w,dT.x,dT.y,dT.z,dR.x,dR.y,dR.z,dR.w" << endl;
		}  

    void run(const sensor_msgs::ImageConstPtr& msg)  
		{   
			if(this->is_calibrated == false)
			{
				// image conversion
				cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC3);
				cv::Mat img = cv_ptr->image;
				// detect
				this->mkd.read(img);
				this->mkd.process();
				if(this->mkd.isDetected && this->cam_info.header.seq > 0)
				//~ if(true)
				{
					circle(img, this->mkd.o, 3, Scalar(0,255,0), 5, 8, 0);
					circle(img, this->mkd.x, 3, Scalar(0,0,255), 5, 8, 0);
					circle(img, this->mkd.y, 3, Scalar(255,0,0), 5, 8, 0);
					circle(img, this->mkd.f, 3, Scalar(255,255,0), 5, 8, 0);
					Mat cameraMatrix = (Mat1d(3, 3) << this->cam_info.K[0], 0, this->cam_info.K[2], 0, this->cam_info.K[4], this->cam_info.K[5], 0, 0, 1);
					Mat distCoeffs = (Mat1d(1, 5) << 0, 0, 0, 0, 0);
					Mat rvec,tvec;
					vector<Point3f> p3d;
					Point3f O = Point3f(0.075, 0.075, 0);
					Point3f X = Point3f(0.075, -0.075, 0);
					Point3f Y = Point3f(-0.075, 0.075, 0);
					Point3f F = Point3f(-0.075, -0.075, 0);
					p3d.push_back(O); p3d.push_back(X); p3d.push_back(Y); p3d.push_back(F);
					vector<Point2f> p2d;
					p2d.push_back(this->mkd.o); p2d.push_back(this->mkd.x); p2d.push_back(this->mkd.y); p2d.push_back(this->mkd.f);
					solvePnP(p3d, p2d, cameraMatrix, distCoeffs, rvec, tvec, true, CV_P3P);
					Mat R;
					Rodrigues(rvec, R);
					R = R.t();
					Mat T = -R * Mat(tvec);
					// change coordinate
					Mat opencv2ros = (Mat1d(3, 3) << 0,-1,0,-1,0,0,0,0,-1);
					R = opencv2ros * R;
					T = opencv2ros * Mat(T);
					double Q[4];
					getQuaternion(R, Q);
					//~ if(true)
					if(this->mocap_o_detected && this->mocap_x_detected && this->mocap_y_detected && this->mocap_cam_detected)
					{
						Vector3d vec_x(this->mocap_o.x-this->mocap_y.x,this->mocap_o.y-this->mocap_y.y,this->mocap_o.z-this->mocap_y.z);
						vec_x = vec_x / vec_x.norm();
						Vector3d vec_y(this->mocap_o.x-this->mocap_x.x,this->mocap_o.y-this->mocap_x.y,this->mocap_o.z-this->mocap_x.z);
						vec_y = vec_y / vec_y.norm();
						Vector3d vec_z = vec_x.cross(vec_y);
						vec_y = vec_z.cross(vec_x);
						vec_y = vec_y / vec_y.norm();
						// marker pose in mocap
						Matrix3d R_marker(3,3);
						R_marker << vec_x(0), vec_x(1), vec_x(2),
									vec_y(0), vec_y(1), vec_y(2),
									vec_z(0), vec_z(1), vec_z(2);
						Vector3d T_marker(3);
						T_marker << this->mocap_o.x, this->mocap_o.y, this->mocap_o.z;
						// cam pose in mocap
						Quaterniond Q_cam_m;
						Q_cam_m.x() = this->mocap_cam.pose.orientation.x;
						Q_cam_m.y() = this->mocap_cam.pose.orientation.y;
						Q_cam_m.z() = this->mocap_cam.pose.orientation.z;
						Q_cam_m.w() = this->mocap_cam.pose.orientation.w;
						Matrix3d R_cam_m = Q_cam_m.normalized().toRotationMatrix();
						Vector3d T_cam_m(3);
						T_cam_m << this->mocap_cam.pose.position.x, this->mocap_cam.pose.position.y, this->mocap_cam.pose.position.z;	
						// cam pose in marker coordinate
						Matrix3d R_cam_c(3,3);
						R_cam_c << 	R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
									R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
									R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
						Vector3d T_cam_c(3);
						T_cam_c << T.at<double>(0,0), T.at<double>(1,0), T.at<double>(2,0);
						// calculate cam pose in world coordinate
						Matrix3d R_cam_w = R_marker.inverse() * R_cam_c;
						Vector3d T_cam_w = T_marker + R_marker.inverse() * T_cam_c;
						// calculate the difference between cam pose in mocap and cam pose from image
						this->dR = (1 - this->alpha) * this->dR + this->alpha * (R_cam_m.inverse() * R_cam_w);
						this->dT = (1 - this->alpha) * this->dT + this->alpha * (T_cam_w - T_cam_m);

						// // sent measured camera pose
						// cout << "----- NEW DETECTION -----" << endl;
						// geometry_msgs::PoseStamped meapose;
						// meapose.header.frame_id = "world";
						// meapose.header.stamp = ros::Time::now();
						// meapose.pose.position.x = T_cam_w(0);
						// meapose.pose.position.y = T_cam_w(1);
						// meapose.pose.position.z = T_cam_w(2);
						// Quaterniond Q_cam_w(R_cam_w);
						// meapose.pose.orientation.x = Q_cam_w.x();
						// meapose.pose.orientation.y = Q_cam_w.y();
						// meapose.pose.orientation.z = Q_cam_w.z();
						// meapose.pose.orientation.w = Q_cam_w.w();
						// meaposepub.publish(meapose);

						// // sent estimated camera pose
						// geometry_msgs::PoseStamped estpose;
						// //~ Matrix3d R_est = R_cam_m;
						// //~ Vector3d T_est = T_cam_m;
						// Matrix3d R_est = R_cam_m * this->dR;
						// Vector3d T_est = T_cam_m + this->dT;
						// estpose.header.frame_id = "world";
						// estpose.header.stamp = ros::Time::now();
						// estpose.pose.position.x = T_est(0);
						// estpose.pose.position.y = T_est(1);
						// estpose.pose.position.z = T_est(2);
						// Quaterniond Q_est(R_est);
						// estpose.pose.orientation.x = Q_est.x();
						// estpose.pose.orientation.y = Q_est.y();
						// estpose.pose.orientation.z = Q_est.z();
						// estpose.pose.orientation.w = Q_est.w();
						// estposepub.publish(estpose);

						// cout << "est: " << T_est << " mea: " << T_cam_w << endl;
						// cout << "est: " << R_est << " mea: " << R_cam_w << endl;

						// store data in file
						// Matrix3d dr = (R_cam_m.inverse() * R_cam_w);
						// Vector3d dt = (T_cam_w - T_cam_m);
						// Quaterniond dq(dr);
						// this->myfile 	<< T_cam_w(0) << ","
						// 				<< T_cam_w(1) << ","
						// 				<< T_cam_w(2) << ","
						// 				<< Q_cam_w.x() << ","
						// 				<< Q_cam_w.y() << ","
						// 				<< Q_cam_w.z() << ","
						// 				<< Q_cam_w.w() << ","
						// 				<< T_est(0) << ","
						// 				<< T_est(1) << ","
						// 				<< T_est(2) << ","
						// 				<< Q_est.x() << ","
						// 				<< Q_est.y() << ","
						// 				<< Q_est.z() << ","
						// 				<< Q_est.w() << ","
						// 				<< dt(0) << ","
						// 				<< dt(1) << ","
						// 				<< dt(2) << ","
						// 				<< dq.x() << ","
						// 				<< dq.y() << ","
						// 				<< dq.z() << ","
						// 				<< dq.w() << "," << endl;
					}
				}
				namedWindow("BW", CV_WINDOW_NORMAL);
				imshow("BW", this->mkd.img);
				namedWindow("RGB", CV_WINDOW_NORMAL);
				imshow("RGB", img);
				waitKey(1);
			}
		}

	void store_cam_info(const sensor_msgs::CameraInfo& msg)  
		{  
			this->cam_info = msg;
		}

	void store_mocap_o(const geometry_msgs::PoseStamped& msg)
		{
			this->mocap_o_detected = true;
			this->mocap_o = Point3f(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);
			cout << "marker_o receied!" << endl;
		}

	void store_mocap_x(const geometry_msgs::PoseStamped& msg)
		{
			this->mocap_x_detected = true;
			this->mocap_x = Point3f(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);
			cout << "marker_x receied!" << endl;
		}

	void store_mocap_y(const geometry_msgs::PoseStamped& msg)
		{
			this->mocap_y_detected = true;
			this->mocap_y = Point3f(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);
			cout << "marker_y receied!" << endl;
		}

	void store_mocap_cam(const geometry_msgs::PoseStamped& msg)
		{
			this->mocap_cam_detected = true;
			this->mocap_cam = msg;
			cout << "mocap_cam receied!" << endl;
			// sent tf in mocap
			tf::Transform transform;
			transform.setOrigin( tf::Vector3(this->mocap_cam.pose.position.x,this->mocap_cam.pose.position.y,this->mocap_cam.pose.position.z) );
			transform.setRotation( tf::Quaternion(this->mocap_cam.pose.orientation.x, this->mocap_cam.pose.orientation.y, this->mocap_cam.pose.orientation.z, this->mocap_cam.pose.orientation.w));
			this->br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "cam_in_mocap"));
			// sent tf in world
			Quaterniond Q_cam_m;
			Q_cam_m.x() = this->mocap_cam.pose.orientation.x;
			Q_cam_m.y() = this->mocap_cam.pose.orientation.y;
			Q_cam_m.z() = this->mocap_cam.pose.orientation.z;
			Q_cam_m.w() = this->mocap_cam.pose.orientation.w;
			Matrix3d R_cam_m = Q_cam_m.normalized().toRotationMatrix();
			Vector3d T_cam_m(3);
			T_cam_m << this->mocap_cam.pose.position.x, this->mocap_cam.pose.position.y, this->mocap_cam.pose.position.z;
			Matrix3d R_est = R_cam_m * this->dR;
			Vector3d T_est = T_cam_m + this->dT;
			Quaterniond Q_est(R_est);
			transform.setOrigin( tf::Vector3(T_est(0), T_est(1), T_est(2)) );
			transform.setRotation( tf::Quaternion(Q_est.x(), Q_est.y(), Q_est.z(), Q_est.w()));
			this->br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "camera_opencv"));
			// send pose in world
			geometry_msgs::PoseStamped estpose;
			estpose.header.frame_id = "world";
			estpose.header.stamp = ros::Time::now();
			estpose.pose.position.x = T_est(0);
			estpose.pose.position.y = T_est(1);
			estpose.pose.position.z = T_est(2);
			estpose.pose.orientation.x = Q_est.x();
			estpose.pose.orientation.y = Q_est.y();
			estpose.pose.orientation.z = Q_est.z();
			estpose.pose.orientation.w = Q_est.w();
			estposepub.publish(estpose);
		}

	void stop(const std_msgs::Bool& msg)
		{
			if(msg.data)
			{
				this->is_calibrated = true;
			}
			else
			{
				this->is_calibrated = false;
			}
			
		}

	private:  
		ros::NodeHandle nh;   		// define node
		image_transport::Subscriber imgsub;		// define subscriber for color image
		ros::Subscriber camsub;
		ros::Publisher estposepub;
		// ros::Publisher meaposepub;
		ros::Subscriber mocap_marker_o_sub;
		ros::Subscriber mocap_marker_x_sub;
		ros::Subscriber mocap_marker_y_sub;
		ros::Subscriber mocap_marker_cam_sub;
		ros::Subscriber bool_sub;
		bool is_calibrated;
		tf::TransformBroadcaster br;
		bool mocap_o_detected;
		bool mocap_x_detected;
		bool mocap_y_detected;
		bool mocap_cam_detected;
		Point3f mocap_o;
		Point3f mocap_x;
		Point3f mocap_y;
		geometry_msgs::PoseStamped mocap_cam;
		Mat image;
		marker_detector mkd;
		sensor_msgs::CameraInfo cam_info;
		float alpha;
		Matrix3d dR;
		Vector3d dT;
		ofstream myfile;
};

int main(int argc, char **argv)  
{  
	//Initiate ROS  
	ros::init(argc, argv, "marker_detector");  

	//Create an object of class SubscribeAndPublish that will take care of everything  
	my_detector SAPObject; 

	ros::spin();  
	return 0;  
} 
