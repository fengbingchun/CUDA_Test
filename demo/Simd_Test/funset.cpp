#include "funset.hpp"
#include <string>
#include <opencv2/opencv.hpp>
#include <Simd/SimdSse2.h>
#include <Simd/SimdBase.h>

int test_bgra2gray()
{
#ifdef _MSC_VER
	std::string strImageName = "../../../test_data/images/lena.png";
#else
	std::string strImageName = "test_data/images/lena.png";
#endif
	int iImageWidth = 10000;
	int iImageHeight = 10000;

	cv::Mat matSrc = cv::imread(strImageName, 1);
	cv::cvtColor(matSrc, matSrc, cv::COLOR_BGR2BGRA);
	cv::resize(matSrc, matSrc, cv::Size(iImageWidth, iImageHeight), 0, 0, 1);

	cv::Mat matDst1, matDst2;
	matDst1 = cv::Mat::zeros(iImageHeight, iImageWidth, CV_8UC1);
	matDst2 = cv::Mat::zeros(iImageHeight, iImageWidth, CV_8UC1);

	int iRemainder = iImageWidth & 0x03;
	int iGrayStride = iRemainder ? iImageWidth + 4 - iRemainder : iImageWidth;
	CV_Assert(iRemainder == 0);

	double dTimeC = cv::getTickCount();
	Simd::Base::BgraToGray(matSrc.data, iImageWidth, iImageHeight, iImageWidth * 4, matDst1.data, iGrayStride);
	dTimeC = ((double)cv::getTickCount() - dTimeC) / cv::getTickFrequency();

	double dTimeSimd = cv::getTickCount();
	Simd::Sse2::BgraToGray(matSrc.data, iImageWidth, iImageHeight, iImageWidth * 4, matDst2.data, iGrayStride);
	dTimeSimd = ((double)cv::getTickCount() - dTimeSimd) / cv::getTickFrequency();

	std::cout << "C run time : " << dTimeC << std::endl;
	std::cout << "Simd run time : " << dTimeSimd << std::endl;

	int iDiffCount = 0;

	for (int i = 0; i < iImageHeight; i++) {
		uchar* p1 = matDst1.ptr<uchar>(i);
		uchar* p2 = matDst2.ptr<uchar>(i);

		for (int j = 0; j < iImageWidth; j++) {
			if (p1[j] != p2[j])
				iDiffCount++;
		}
	}

	std::cout << "the different count: " << iDiffCount << std::endl;
	return 0;
}
