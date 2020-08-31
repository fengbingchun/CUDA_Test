#ifndef ASSEMBLY_LANGUAGE_TEST_COMMON_HPP_
#define ASSEMBLY_LANGUAGE_TEST_COMMON_HPP_

#include <iostream>
#include <iomanip>

extern "C" void asmMain();

extern "C" void _printFloat(float f)
{
	std::cout << std::setprecision(7) << f << std::endl;
}

extern "C" void _printDouble(double f)
{
	std::cout << std::setprecision(15) << f << std::endl;
}

#endif // ASSEMBLY_LANGUAGE_TEST_COMMON_HPP_
