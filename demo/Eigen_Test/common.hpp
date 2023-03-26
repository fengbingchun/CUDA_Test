#ifndef FBC_EIGEN_TEST_COMMON_HPP_
#define FBC_EIGEN_TEST_COMMON_HPP_

#include <iostream>

template<typename _Tp>
void print_matrix(const _Tp* data, const int rows, const int cols)
{
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			fprintf(stderr, "  %f  ", static_cast<float>(data[y * cols + x]));
		}
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");
}

#endif // FBC_EIGEN_TEST_COMMON_HPP_
