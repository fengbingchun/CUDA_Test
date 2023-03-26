#include "funset.hpp"
#include <random>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include "common.hpp"

int test_image_process_laplacian()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/79321200
#ifdef __linux__
	cv::Mat src = cv::imread("test_data/images/lena.png", 0);
#else
	cv::Mat src = cv::imread("../../../test_data/images/lena.png", 0);
#endif
	if (!src.data || src.channels() != 1) {
		fprintf(stderr, "read image fail\n");
		return -1;
	}
	int width{ 400 }, height{ 400 };
	cv::resize(src, src, cv::Size(width, height));

	std::unique_ptr<unsigned char[]> data1(new unsigned char[width * height]), data2(new unsigned char[width * height]);
	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds
	int ksize{ 1 };

	CHECK(laplacian_cpu(src.data, width, height, ksize, data1.get(), &elapsed_time1) == 0);
	//CHECK(laplacian_gpu(src.data, width, height, data2.get(), &elapsed_time2) == 0);

	//fprintf(stdout, "gray image edge detection: laplacian: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	cv::Mat dst;
	cv::Laplacian(src, dst, src.depth(), ksize);
#ifdef __linux__
	cv::imwrite("test_data/images/laplacian.png", dst);
#else
	cv::imwrite("../../../test_data/images/laplacian.png", dst);
#endif

	CHECK(compare_result(data1.get(), dst.data, width*height) == 0);
	//CHECK(compare_result(data1.get(), data2.get(), width*height) == 0);

#ifdef __linux__
	save_image(src, dst, width, height / 2, "test_data/images/laplacian_result.png");
#else
	save_image(src, dst, width, height / 2, "../../../test_data/images/laplacian_result.png");
#endif

	return 0;
}

int test_image_process_histogram_equalization()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/79188021
#ifdef __linux__
	const std::string image_name{ "test_data/images/lena.png" };
#else
	const std::string image_name{ "../../../test_data/images/lena.png" };
#endif
	cv::Mat mat = cv::imread(image_name, 0);
	CHECK(mat.data);

	const int width{ mat.cols/*1513*/ }, height{ mat.rows/*1473*/ };
	cv::resize(mat, mat, cv::Size(width, height));

	std::unique_ptr<unsigned char[]> data1(new unsigned char[width * height]), data2(new unsigned char[width * height]);
	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds

	CHECK(histogram_equalization_cpu(mat.data, width, height, data1.get(), &elapsed_time1) == 0);
	//CHECK(histogram_equalization_gpu(mat.data, width, height, data2.get(), &elapsed_time2) == 0);

	//fprintf(stdout, "image histogram equalization: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	cv::Mat dst;
	cv::equalizeHist(mat, dst);
#ifdef __linux__
	cv::imwrite("test_data/images/histogram_equalization.png", dst);
#else
	cv::imwrite("../../../test_data/images/histogram_equalization.png", dst);
#endif

	CHECK(compare_result(data1.get(), dst.data, width*height) == 0);
	//CHECK(compare_result(data1.get(), data2.get(), width*height) == 0);

#ifdef __linux__
	save_image(mat, dst, width, height/2, "test_data/images/histogram_equalization_result.png");
#else
	save_image(mat, dst, width, height/2, "../../../test_data/images/histogram_equalization_result.png");
#endif

	return 0;
}

int test_image_process_bgr2bgr565()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/78995720
#ifdef __linux__
	const std::string image_name{ "test_data/images/lena.png" };
#else
	const std::string image_name{ "../../../test_data/images/lena.png" };
#endif
	cv::Mat mat = cv::imread(image_name, 1);
	CHECK(mat.data);

	const int width{ 1513 }, height{ 1473 };
	cv::resize(mat, mat, cv::Size(width, height));

	std::unique_ptr<unsigned char[]> data1(new unsigned char[width * height * 2]), data2(new unsigned char[width * height * 2]);
	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds

	cv::Mat bgr565;
	cv::cvtColor(mat, bgr565, cv::COLOR_BGR2BGR565);

	CHECK(bgr2bgr565_cpu(mat.data, width, height, data1.get(), &elapsed_time1) == 0);
	CHECK(bgr2bgr565_gpu(mat.data, width, height, data2.get(), &elapsed_time2) == 0);

	fprintf(stdout, "image bgr to bgr565: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	CHECK(compare_result(data1.get(), bgr565.data, width*height * 2) == 0);
	CHECK(compare_result(data1.get(), data2.get(), width*height*2) == 0);

	return 0;
}

int test_image_process_bgr2gray()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/78821765
#ifdef __linux__
	const std::string image_name{ "test_data/images/lena.png" };
#else
	const std::string image_name{ "../../../test_data/images/lena.png" };
#endif
	cv::Mat mat = cv::imread(image_name);
	CHECK(mat.data);

	const int width{ 1513 }, height{ 1473 };
	cv::resize(mat, mat, cv::Size(width, height));

	std::unique_ptr<unsigned char[]> data1(new unsigned char[width * height]), data2(new unsigned char[width * height]);
	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds

	CHECK(bgr2gray_cpu(mat.data, width, height, data1.get(), &elapsed_time1) == 0);
	CHECK(bgr2gray_gpu(mat.data, width, height, data2.get(), &elapsed_time2) == 0);

	cv::Mat dst(height, width, CV_8UC1, data1.get());
#ifdef __linux__
	cv::imwrite("test_data/images/bgr2gray_cpu.png", dst);
#else
	cv::imwrite("../../../test_data/images/bgr2gray_cpu.png", dst);
#endif
	cv::Mat dst2(height, width, CV_8UC1, data2.get());
#ifdef __linux__
	cv::imwrite("test_data/images/bgr2gray_gpu.png", dst2);
#else
	cv::imwrite("../../../test_data/images/bgr2gray_gpu.png", dst2);
#endif
	fprintf(stdout, "image bgr to gray: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	CHECK(compare_result(data1.get(), data2.get(), width*height) == 0);

	return 0;
}

int test_layer_prior_vbox()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/77850422
	std::vector<float> vec1{423.f, 245.f, 1333.f, 1444.f, 123.f, 23.f, 32.f, 66.f};
	std::vector<float> vec2(vec1[6]);
	std::vector<float> vec3(4);
	int length = int(vec1[0] * vec1[1] * vec1[6] * 4 * 2);

	std::unique_ptr<float[]> data1(new float[length]), data2(new float[length]);
	std::for_each(data1.get(), data1.get() + length, [](float& n) {n = 0.f; });
	std::for_each(data2.get(), data2.get() + length, [](float& n) {n = 0.f; });
	generator_random_number(vec2.data(), vec2.size(), 10.f, 100.f);
	generator_random_number(vec3.data(), vec3.size(), 1.f, 10.f);

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds
	int ret = layer_prior_vbox_cpu(data1.get(), length, vec1, vec2, vec3, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(layer_prior_vbox_cpu);

	ret = layer_prior_vbox_gpu(data2.get(), length, vec1, vec2, vec3, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(layer_prior_vbox_gpu);

	compare_result(data1.get(), data2.get(), length);

	fprintf(stderr, "test layer prior vbox: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

int test_layer_reverse()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/77160872
#ifdef __linux__
	std::string image_name{ "test_data/images/lena.png" };
#else
	std::string image_name{ "../../../test_data/images/lena.png" };
#endif
	cv::Mat matSrc = cv::imread(image_name);
	CHECK(matSrc.data);

	cv::cvtColor(matSrc, matSrc, cv::COLOR_BGR2GRAY);
	const int width{ 1511 }, height{ 1473 };
	const auto length = width * height;
	cv::resize(matSrc, matSrc, cv::Size(width, height));
	cv::Mat matTmp1;
	matSrc.convertTo(matTmp1, CV_32FC1);

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds
	const std::vector<int> vec{ 5, 7};
	std::unique_ptr<float[]> dst1(new float[length]), dst2(new float[length]);
	std::for_each(dst1.get(), dst1.get() + length, [](float& n) {n = 0.f; });
	std::for_each(dst2.get(), dst2.get() + length, [](float& n) {n = 0.f; });

	int ret = layer_reverse_cpu((float*)matTmp1.data, dst1.get(), length, vec, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(image_reverse_cpu);

	ret = layer_reverse_gpu((float*)matTmp1.data, dst2.get(), length, vec, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(image_reverse_gpu);

	compare_result(dst1.get(), dst2.get(), length);

	cv::Mat matTmp2(height, width, CV_32FC1, dst2.get()), matDst;
	matTmp2.convertTo(matDst, CV_8UC1);

#ifdef __linux__
	save_image(matSrc, matDst, 400, 200, "test_data/images/image_reverse.png");
#else
	save_image(matSrc, matDst, 400, 200, "../../../test_data/images/image_reverse.png");
#endif

	fprintf(stderr, "test layer reverse: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

int test_layer_channel_normalize()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/76976024
#ifdef __linux__
	std::string image_name{ "test_data/images/lena.png" };
#else
	std::string image_name{ "../../../test_data/images/lena.png" };
#endif
	cv::Mat matSrc = cv::imread(image_name);
	if (!matSrc.data) {
		fprintf(stderr, "read image fail: %s\n", image_name.c_str());
		return -1;
	}

	const int width{ 511 }, height{ 473 }, channels{ 3 };
	cv::resize(matSrc, matSrc, cv::Size(width, height));
	matSrc.convertTo(matSrc, CV_32FC3);
	std::vector<cv::Mat> matSplit;
	cv::split(matSrc, matSplit);
	CHECK(matSplit.size() == channels);
	std::unique_ptr<float[]> data(new float[matSplit[0].cols * matSplit[0].rows * channels]);
	size_t length{ matSplit[0].cols * matSplit[0].rows * sizeof(float) };
	for (int i = 0; i < channels; ++i) {
		memcpy(data.get() + matSplit[0].cols * matSplit[0].rows * i, matSplit[i].data, length);
	}

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds
	std::unique_ptr<float[]> dst1(new float[matSplit[0].cols * matSplit[0].rows * channels]);
	std::unique_ptr<float[]> dst2(new float[matSplit[0].cols * matSplit[0].rows * channels]);

	int ret = layer_channel_normalize_cpu(data.get(), dst1.get(), width, height, channels, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(image_normalize_cpu);

	ret = layer_channel_normalize_gpu(data.get(), dst2.get(), width, height, channels, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(image_normalize_gpu);

	int count{ 0 }, num{ width * height * channels };
	for (int i = 0; i < num; ++i) {
		if (fabs(dst1[i] - dst2[i]) > 0.01/*EPS_*/) {
			fprintf(stderr, "index: %d, val1: %f, val2: %f\n", i, dst1[i], dst2[i]);
			++count;
		}
		if (count > 100) return -1;
	}

	std::vector<cv::Mat> merge(channels);
	for (int i = 0; i < channels; ++i) {
		merge[i] = cv::Mat(height, width, CV_32FC1, dst2.get() + i * width * height);
	}
	cv::Mat dst3;
	cv::merge(merge, dst3);
	dst3.convertTo(dst3, CV_8UC3, 255.f);
#ifdef __linux__
	cv::imwrite("test_data/images/image_normalize.png", dst3);
#else
	cv::imwrite("../../../test_data/images/image_normalize.png", dst3);
#endif
	//cv::resize(matSrc, matSrc, cv::Size(width, height));
#ifdef __linux__
	//cv::imwrite("test_data/images/image_src.png", matSrc);
#else
	//cv::imwrite("../../../test_data/images/image_src.png", matSrc);
#endif

	fprintf(stderr, "test layer channel normalize: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

int test_get_device_info()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/76902556
	int ret = get_device_info();
	if (ret != 0) PRINT_ERROR_INFO(get_device_info);

	return 0;
}

int test_matrix_mul()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/76618165
	// Matrix multiplication: C = A * B
	// 矩阵A、B的宽、高应是32的整数倍
	const int rowsA{ 352 }, colsA{ 672 }, rowsB = colsA, colsB{ 384 };
	std::unique_ptr<float[]> A(new float[colsA*rowsA]);
	std::unique_ptr<float[]> B(new float[colsB*rowsB]);
	std::unique_ptr<float[]> C1(new float[rowsA*colsB]);
	std::unique_ptr<float[]> C2(new float[rowsA*colsB]);

	generator_random_number(A.get(), colsA*rowsA, -1.f, 1.f);
	generator_random_number(B.get(), colsB*rowsB, -1.f, 1.f);

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds
	int ret = matrix_mul_cpu(A.get(), B.get(), C1.get(), colsA, rowsA, colsB, rowsB, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(matrix_mul_cpu);

	ret = matrix_mul_gpu(A.get(), B.get(), C2.get(), colsA, rowsA, colsB, rowsB, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(matrix_mul_gpu);

	int count{ 0 };
	for (int i = 0; i < rowsA*colsB; ++i) {
		if (count > 100) return -1;
		if (fabs(C1[i] - C2[i]) > EPS_) {
			fprintf(stderr, "Result verification failed at element %d, C1: %f, C2: %f\n",
				i, C1[i], C2[i]);
			++count;
		}
	}

	fprintf(stderr, "test matrix mul: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

int test_dot_product()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/75669764
	//       http://blog.csdn.net/fengbingchun/article/details/76571955
	const int length{ 1024 * 1024 * 33 };
	std::unique_ptr<float[]> A(new float[length]);
	std::unique_ptr<float[]> B(new float[length]);

	generator_random_number(A.get(), length, -10.f, 10.f);
	generator_random_number(B.get(), length, -10.f, 10.f);

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds
	float value1{ 0.f }, value2{ 0.f };

	int ret = dot_product_cpu(A.get(), B.get(), &value1, length, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(long_vector_add_cpu);

	ret = dot_product_gpu(A.get(), B.get(), &value2, length, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(matrix_mul_gpu);

	if (fabs(value1 - value2) > EPS_) {
		fprintf(stderr, "Result verification failed value1: %f, value2: %f\n", value1, value2);
	}

	fprintf(stderr, "test dot product: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

int test_streams()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/76532198
	const int length{ 1024 * 1024 * 20};
	std::unique_ptr<int[]> A(new int[length]);
	std::unique_ptr<int[]> B(new int[length]);
	std::unique_ptr<int[]> C1(new int[length]);
	std::unique_ptr<int[]> C2(new int[length]);

	generator_random_number<int>(A.get(), length, -100, 100);
	generator_random_number<int>(B.get(), length, -100, 100);
	std::for_each(C1.get(), C1.get() + length, [](int& n) {n = 0; });
	std::for_each(C2.get(), C2.get() + length, [](int& n) {n = 0; });

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds

	int ret = streams_cpu(A.get(), B.get(), C1.get(), length, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(streams_cpu);

	ret = streams_gpu(A.get(), B.get(), C2.get(), length, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(streams_gpu);

	for (int i = 0; i < length; ++i) {
		if (C1[i] != C2[i]) {
			fprintf(stderr, "their values are different at: %d, val1: %d, val2: %d\n",
				i, C1[i], C2[i]);
			return -1;
		}
	}

	fprintf(stderr, "test streams' usage: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

int test_calculate_histogram()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/76417598
	const int length{ 10 * 1024 * 1024 }; // 100MB
	std::unique_ptr<unsigned char[]> data(new unsigned char[length]);
	generator_random_number<unsigned char>(data.get(), length, 0, 255);

	const int hist_size{ 256 };
	std::unique_ptr<unsigned int[]> hist1(new unsigned int[hist_size]), hist2(new unsigned int[hist_size]);
	std::for_each(hist1.get(), hist1.get() + hist_size, [](unsigned int& n) {n = 0; });
	std::for_each(hist2.get(), hist2.get() + hist_size, [](unsigned int& n) {n = 0; });

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds
	unsigned int value1{ 0 }, value2{ 0 };

	int ret = calculate_histogram_cpu(data.get(), length, hist1.get(), value1, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(calculate_histogram_cpu);

	ret = calculate_histogram_gpu(data.get(), length, hist2.get(), value2, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(calculate_histogram_gpu);

	if (value1 != value2) {
		fprintf(stderr, "their values are different: val1: %d, val2: %d\n", value1, value2);
		return -1;
	}
	for (int i = 0; i < hist_size; ++i) {
		if (hist1[i] != hist2[i]) {
			fprintf(stderr, "their values are different at: %d, val1: %d, val2: %d\n",
				i, hist1[i], hist2[i]);
			return -1;
		}
	}

	fprintf(stderr, "test calculate histogram: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

int test_heat_conduction()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/76228257
	const int width{ 1024 }, height = width;
	cv::Mat mat1(height, width, CV_8UC4), mat2(height, width, CV_8UC4);

	const float speed{ 0.25f }, max_temp{ 1.f }, min_temp{0.0001f};
	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds

	// intialize the constant data
	std::unique_ptr<float[]> temp(new float[width * height]);
	for (int i = 0; i < width*height; ++i) {
		temp[i] = 0;
		int x = i % width;
		int y = i / height;
		if ((x>300) && (x<600) && (y>310) && (y<601))
			temp[i] = max_temp;
	}

	temp[width * 100 + 100] = (max_temp + min_temp) / 2;
	temp[width * 700 + 100] = min_temp;
	temp[width * 300 + 300] = min_temp;
	temp[width * 200 + 700] = min_temp;

	for (int y = 800; y < 900; ++y) {
		for (int x = 400; x < 500; ++x) {
			temp[x + y * width] = min_temp;
		}
	}

	int ret = heat_conduction_cpu(mat1.data, width, height, temp.get(), speed, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(heat_conduction_cpu);

	ret = heat_conduction_gpu(mat2.data, width, height, temp.get(), speed, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(heat_conduction_gpu);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			cv::Vec4b val1 = mat1.at<cv::Vec4b>(y, x);
			cv::Vec4b val2 = mat2.at<cv::Vec4b>(y, x);

			for (int i = 0; i < 4; ++i) {
				if (val1[i] != val2[i]) {
					fprintf(stderr, "their values are different at (%d, %d), i: %d, val1: %d, val2: %d\n",
						x, y, i, val1[i], val2[i]);
					//return -1;
				}
			}
		}
	}

#ifdef __linux__
	std::string save_image_name{ "test_data/heat_conduction.jpg" };
#else
	std::string save_image_name{ "../../../heat_conduction.jpg" };
#endif
	cv::resize(mat2, mat2, cv::Size(width / 2, height / 2), 0.f, 0.f, 2);
	cv::imwrite(save_image_name, mat2);

	fprintf(stderr, "test heat conduction: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

int test_ray_tracking()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/76135606
	const int spheres{ 20 };
	std::unique_ptr<float[]> A(new float[spheres * 3]);
	std::unique_ptr<float[]> B(new float[spheres * 3]);
	std::unique_ptr<float[]> C(new float[spheres]);

	generator_random_number(A.get(), spheres * 3, 0.f, 1.f);
	generator_random_number(B.get(), spheres * 3, -400.f, 400.f);
	generator_random_number(C.get(), spheres, 20.f, 120.f);

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds

	const int width{ 512 }, height = width;
	cv::Mat mat1(height, width, CV_8UC4), mat2(height, width, CV_8UC4);

	int ret = ray_tracking_cpu(A.get(), B.get(), C.get(), spheres, mat1.data, width, height, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(ray_tracking_cpu);

	ret = ray_tracking_gpu(A.get(), B.get(), C.get(), spheres, mat2.data, width, height, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(ray_tracking_gpu);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			cv::Vec4b val1 = mat1.at<cv::Vec4b>(y, x);
			cv::Vec4b val2 = mat2.at<cv::Vec4b>(y, x);

			for (int i = 0; i < 4; ++i) {
				if (val1[i] != val2[i]) {
					fprintf(stderr, "their values are different at (%d, %d), i: %d, val1: %d, val2: %d\n",
						x, y, i, val1[i], val2[i]);
					//return -1;
				}
			}
		}
	}

#ifdef __linux__
	const std::string save_image_name{ "test_data/ray_tracking.jpg" };
#else
	const std::string save_image_name{ "../../../ray_tracking.jpg" };
#endif
	cv::imwrite(save_image_name, mat2);

	fprintf(stderr, "ray tracking: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

int test_green_ball()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/76058122
	const int width{ 512 }, height = width;
	cv::Mat mat1(height, width, CV_8UC4), mat2(height, width, CV_8UC4);

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds

	int ret = green_ball_cpu(mat1.data, width, height, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(green_ball_cpu);

	ret = green_ball_gpu(mat2.data, width, height, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(green_ball_gpu);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			cv::Vec4b val1 = mat1.at<cv::Vec4b>(y, x);
			cv::Vec4b val2 = mat2.at<cv::Vec4b>(y, x);

			for (int i = 0; i < 4; ++i) {
				if (val1[i] != val2[i]) {
					fprintf(stderr, "their values are different at (%d, %d), i: %d, val1: %d, val2: %d\n",
						x, y, i, val1[i], val2[i]);
					//return -1;
				}
			}
		}
	}

#ifdef __linux__
	const std::string save_image_name{ "test_data/gree_ball.jpg" };
#else
	const std::string save_image_name{ "../../../gree_ball.jpg" };
#endif
	cv::imwrite(save_image_name, mat2);

	fprintf(stderr, "test green ball: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

int test_ripple()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/76043190
	const int width{ 512 }, height = width;
	const int ticks{ 999 };
	cv::Mat mat1(height, width, CV_8UC4), mat2(height, width, CV_8UC4);

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds

	int ret = ripple_cpu(mat1.data, width, height, ticks, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(ripple_cpu);

	ret = ripple_gpu(mat2.data, width, height, ticks, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(ripple_gpu);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			cv::Vec4b val1 = mat1.at<cv::Vec4b>(y, x);
			cv::Vec4b val2 = mat2.at<cv::Vec4b>(y, x);

			for (int i = 0; i < 4; ++i) {
				if (val1[i] != val2[i]) {
					fprintf(stderr, "their values are different at (%d, %d), i: %d, val1: %d, val2: %d\n",
						x, y, i, val1[i], val2[i]);
					return -1;
				}
			}
		}
	}

#ifdef __linux__
	const std::string save_image_name{ "test_data/ripple.jpg" };
#else
	const std::string save_image_name{ "../../../ripple.jpg" };
#endif
	cv::imwrite(save_image_name, mat2);

	fprintf(stderr, "cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

int test_julia()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/76020154
	const int width{ 512 }, height = width;
	const float scale{ 1.5f };
	cv::Mat mat1(height, width, CV_8UC4), mat2(height, width, CV_8UC4);

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds

	int ret = julia_cpu(mat1.data, width, height, scale, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(julia_cpu);

	ret = julia_gpu(mat2.data, width, height, scale, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(julia_gpu);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			cv::Vec4b val1 = mat1.at<cv::Vec4b>(y, x);
			cv::Vec4b val2 = mat2.at<cv::Vec4b>(y, x);

			for (int i = 0; i < 4; ++i) {
				if (val1[i] != val2[i]) {
					fprintf(stderr, "their values are different at (%d, %d), i: %d, val1: %d, val2: %d\n",
						x, y, i, val1[i], val2[i]);
					//return -1;
				}
			}
		}
	}

#ifdef __linux__
	const std::string save_image_name{ "test_data/julia.jpg" };
#else
	const std::string save_image_name{ "../../../julia.jpg" };
#endif
	cv::imwrite(save_image_name, mat2);

	fprintf(stderr, "cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

int test_long_vector_add()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/75570546
	const int length{ 100000000 };
	std::unique_ptr<float[]> A(new float[length]);
	std::unique_ptr<float[]> B(new float[length]);
	std::unique_ptr<float[]> C1(new float[length]);
	std::unique_ptr<float[]> C2(new float[length]);

	generator_random_number(A.get(), length, -1000.f, 1000.f);
	generator_random_number(B.get(), length, -1000.f, 1000.f);

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds
	int ret = long_vector_add_cpu(A.get(), B.get(), C1.get(), length, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(long_vector_add_cpu);

	ret = long_vector_add_gpu(A.get(), B.get(), C2.get(), length, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(matrix_mul_gpu);

	int count_error{ 0 };
	for (int i = 0; i < length; ++i) {
		if (count_error > 100) return -1;
		if (fabs(C1[i] - C2[i]) > EPS_) {
			fprintf(stderr, "Result verification failed at element %d, C1: %f, C2: %f\n",
				i, C1[i], C2[i]);
			++count_error;
		}
	}

	fprintf(stderr, "cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

int test_vector_add()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/74120057
	// Vector addition: C = A + B, implements element by element vector addition
	const int numElements{ 50000 };
	std::vector<float> A(numElements), B(numElements), C1(numElements), C2(numElements);

	// Initialize vector
	for (int i = 0; i < numElements; ++i) {
		A[i] = rand() / (float)RAND_MAX;
		B[i] = rand() / (float)RAND_MAX;
	}

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds
	int ret = vector_add_cpu(A.data(), B.data(), C1.data(), numElements, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(vectorAdd_cpu);

	ret = vector_add_gpu(A.data(), B.data(), C2.data(), numElements, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(vectorAdd_gpu);

	for (int i = 0; i < numElements; ++i) {
		if (fabs(C1[i] - C2[i]) > EPS_) {
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			return -1;
		}
	}

	fprintf(stderr, "cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

