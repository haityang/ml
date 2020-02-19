#include "opencv2/opencv.hpp"
//#include "opencv2/contrib/contrib.hpp"

using namespace cv;
using namespace cv::ml;

int main(int argc, char **argv)
{
	const int kWidth = 512; 
	const int kHeight = 512; 
	Vec3b red(0, 0, 255), green(0, 255, 0), blue(255, 0, 0);
	
	Mat image = Mat::zeros(kHeight, kWidth, CV_8UC3);  
	
	int labels[30];
	for (int i  = 0 ; i < 10; i++)
		labels[i] = 1; 
	for (int i = 10; i < 20; i++)
		labels[i] = 2; 
	for (int i = 20; i < 30; i++)
		labels[i] = 3; 
	Mat trainResponse(30, 1, CV_32SC1, labels);
	
	float trainDataArray[30][2];
	RNG rng; 
	for (int i = 0; i < 10; i++)
	{
		trainDataArray[i][0] = 250 + static_cast<float>(rng.gaussian(30));
		trainDataArray[i][1] = 250 + static_cast<float>(rng.gaussian(30));
	}
	for (int i = 10; i < 20; i++)
	{
		trainDataArray[i][0] = 150 + static_cast<float>(rng.gaussian(30));
		trainDataArray[i][1] = 150 + static_cast<float>(rng.gaussian(30));
	}
	for (int i = 20; i < 30; i++)
	{
		trainDataArray[i][0] = 320 + static_cast<float>(rng.gaussian(30));
		trainDataArray[i][1] = 150 + static_cast<float>(rng.gaussian(30));
	}
	Mat trainData(30, 2, CV_32FC1, trainDataArray);
	Ptr<NormalBayesClassifier> model=NormalBayesClassifier::create();
	Ptr<TrainData> tData =TrainData::create(trainData, ROW_SAMPLE, trainResponse);
	model->train(tData);
	
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{

			Mat sampleMat = (Mat_<float>(1, 2) << j, i); 	
			float response = model->predict(sampleMat); 
			if (response == 1)
				image.at<Vec3b>(i, j) = red;
			else if (response == 2)
				image.at<Vec3b>(i, j) = green;
			else
				image.at<Vec3b>(i, j) = blue;
		}
	}
	
	for (int i = 0; i < trainData.rows; i++)
	{
		const float* v = trainData.ptr<float>(i);
		Point pt = Point((int)v[0], (int)v[1]);
		if (labels[i] == 1)
			circle(image, pt, 5, Scalar::all(0), -1, 8); 
		else if (labels[i] == 2)
			circle(image, pt, 5, Scalar::all(128), -1, 8);
		else
			circle(image, pt, 5, Scalar::all(255), -1, 8);
	}

	imshow("Bayessian classifier demo", image);
	waitKey(0);
	return 0;
}
