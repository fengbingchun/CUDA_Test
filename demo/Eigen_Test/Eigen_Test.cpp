#include <iostream>
#include "funset.hpp"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
	int ret = test_quaternion();

	if (ret == 0) fprintf(stdout, "========== test success ==========\n");
	else fprintf(stderr, "********** test fail: %d **********\n", ret);

	return 0;
}

