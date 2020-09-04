#include "../funset.hpp"
#include <iostream>
#include <x86intrin.h>
#include <math.h>

// from: https://github.com/brianrhall/Assembly/blob/master/Chapter_9/Program%209.1/Program_9.1a_Clang_GCC.cpp
int test_inline_int()
{
	int var1 = 1234, var2;
	asm ("mov %1,%%eax \n\t"
		"add $2,%%eax \n\t"
		"mov %%eax,%0 \n\t"
		:"=r" (var2)       /* %0: Out */
		:"r" (var1)        /* %1: In */
		:"%eax"            /* Overwrite */
       );

	fprintf(stdout, "var2: %d\n", var2);
	return 0;
}

// from: https://github.com/brianrhall/Assembly/blob/master/Chapter_9/Program%209.2/Program_9.2b_Clang_GCC.cpp
int test_inline_float()
{
	// square root of PI example
	const double PI = M_PI;                 // needs to be a const double
	const double* ptr_PI = &PI;             // get the address of PI
	__m128d vecPI = _mm_load_sd(ptr_PI);    // load PI into a vector
	__m128d sqrtPI = _mm_sqrt_pd(vecPI);    // get square root
	fprintf(stdout, "value: %f\n", sqrtPI[0]);

	// SSE scalar example, adding a scalar value to itself
	float var = 1.2;
	float* ptr_var = &var;
	__m128 varVec = _mm_load_ss(ptr_var);
	varVec = _mm_add_ss(varVec, varVec);
	fprintf(stdout, "value: %f\n", varVec[0]);

	return 0;
}
