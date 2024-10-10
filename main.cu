#include "test.cuh"
#include "convolutions.cuh"
#include "grayscale.cuh"
#include "filters.cuh"
#include "kernels.hpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#define WIDTH 640
#define HEIGHT 480

int main() {

	cv::Mat src;

	cv::VideoCapture camera(0);
	camera.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, WIDTH);
	camera.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, HEIGHT);

	camera.read(src);
	cv::Mat gray = cv::Mat(src.rows, src.cols, CV_8UC1);
	cv::Mat gauss = cv::Mat(HEIGHT, WIDTH, CV_8UC3);
	cv::Mat gauss_gray = cv::Mat(HEIGHT, WIDTH, CV_8UC1);

	if(!camera.isOpened())
	{
		printf("Camera could not be opened\n");
		printf("Please change device number\n");
		return -1;
	}


	while(true)
	{
		camera.read(src);
		cv::imshow("Source", src);

		/**
		gpu::grayscale_avg_3ch_1ch(src.data, gray.data, src.cols, src.rows);
		cv::imshow("Grayscale", gray);
		**/

		gpu::conv_3ch_2d_constant(src.data, gauss.data, src.cols, src.rows, GAUS_KERNEL_3x3, 3, 3);
		cv::imshow("Gaussian Blur", gauss);

		/**
		gpu::conv(src.data, gauss_gray.data, src.cols, src.rows, GAUS_KERNEL_3x3, 3, 3);
		cv::imshow("Gaussian Blur", gauss);
		**/

		if (cv::waitKey(5) == 27)
				return 0;

	}

  return 0;
}
