// KazeOpenCV.cpp : 定义控制台应用程序的入口点。
//

#include "predep.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "KAZE/kaze_features.h"

#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )
#pragma comment( lib, cvLIB("flann") )
#pragma comment( lib, cvLIB("features2d") )
#pragma comment( lib, cvLIB("calib3d") )


using namespace std;
using namespace cv;


int main(int argc, char** argv[])
{
	Mat img_1 = imread("box.png");
	Mat img_2 = imread("box_in_scene.png");

	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	toptions opt;
	opt.extended = true;		// 1 - 128-bit vector, 0 - 64-bit vector, default: 0
	opt.verbosity = true;		// 1 - show detail information while caculating KAZE, 0 - unshow, default: 0

	KAZE detector_1(opt);
	KAZE detector_2(opt);

	double t2 = 0.0, t1 = 0.0, tkaze = 0.0;
	int64 start_t1 = cv::getTickCount();

	//-- Detect keypoints and calculate descriptors
	detector_1(img_1, keypoints_1, descriptors_1);
	detector_2(img_2, keypoints_2, descriptors_2);

	t2 = cv::getTickCount();
	tkaze = 1000.0 * (t2 - start_t1) / cv::getTickFrequency();

	cout << "\n\n-- Total detection time (ms): " << tkaze << endl;
	printf("-- Keypoint number of img_1 : %d \n", keypoints_1.size() );
	printf("-- Keypoint number of img_2 : %d \n", keypoints_2.size() );

	//-- Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );
	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_1.rows; i++ )
	{ 
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	//-- Find initial good matches (i.e. whose distance is less than 2*min_dist )
	vector< DMatch > good_matches, inliers;
	for( int i = 0; i < descriptors_1.rows; i++ )
	{ 
		if( matches[i].distance < 2*min_dist )	
		{ 
			good_matches.push_back( matches[i]); 
		}
	}

	cout << "-- Computing homography (RANSAC)..." << endl;
	//-- Get the keypoints from the good matches
	vector<Point2f> points1( good_matches.size() ); 
	vector<Point2f> points2( good_matches.size() ); 
	for( size_t i = 0; i < good_matches.size(); i++ )
	{
		points1[i] = keypoints_1[ good_matches[i].queryIdx ].pt;
		points2[i] = keypoints_2[ good_matches[i].trainIdx ].pt;
	}

	//-- Computing homography (RANSAC) and find inliers
	vector<uchar> flags(points1.size(), 0);
	Mat H = findHomography( points1, points2, CV_RANSAC, 3.0, flags );
	//cout << H << endl << endl;
	for (int i = 0; i < good_matches.size(); i++)
	{
		if (flags[i])
		{
			inliers.push_back( good_matches[i] );
		}
	}

	//-- Draw Keypoints
	Mat img_1k, img_2k;
	drawKeypoints(img_1, keypoints_1, img_1k, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img_2, keypoints_2, img_2k, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//-- Draw inliers
	Mat img_matches;
	drawMatches( img_1, keypoints_1, img_2, keypoints_2,
		inliers, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	printf("-- Number of Matches : %d \n", good_matches.size() );
	printf("-- Number of Inliers : %d \n", inliers.size() );
	printf("-- Match rate : %f \n", inliers.size() / (float)good_matches.size() );

	//-- Localize the object
	//-- Get the corners from the image_1 ( the object to be "detected" )
	vector<Point2f> obj_corners;
	obj_corners.push_back( Point2f(0,0) );
	obj_corners.push_back( Point2f(img_1.cols,0) );
	obj_corners.push_back( Point2f(img_1.cols,img_1.rows) );
	obj_corners.push_back( Point2f(0,img_1.rows) );

	if (!H.empty())
	{
		vector<Point2f> scene_corners;
		perspectiveTransform(obj_corners, scene_corners, H);

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		int npts = scene_corners.size();
		for (int i=0; i<npts; i++)
			line( img_matches, scene_corners[i] + Point2f( img_1.cols, 0), 
				scene_corners[(i+1)%npts] + Point2f( img_1.cols, 0), Scalar(0,0,255), 2 );
	}

	//-- Show detected matches
	cout << "-- Show detected matches." << endl;
	namedWindow("Image 1",CV_WINDOW_NORMAL);
	namedWindow("Image 2",CV_WINDOW_NORMAL);
	namedWindow("Good Matches",CV_WINDOW_NORMAL);
	imshow( "Image 1", img_1k );
	imshow( "Image 2", img_2k );
	imshow( "Good Matches", img_matches );
	waitKey(0);
	destroyAllWindows();

	return 0;
}

