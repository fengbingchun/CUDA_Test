#include "../funset.hpp"
#include <intrin.h>
#include <iostream>

// from: https://github.com/brianrhall/Assembly/blob/master/Chapter_9/Program%209.1/Program_9.1b_VisualC.cpp
int test_inline_int()
{
	__m128i var1 = _mm_cvtsi32_si128(1234);
	__m128i var2 = _mm_cvtsi32_si128(2);

	var1 = _mm_add_epi64(var1, var2);

	int output = _mm_cvtsi128_si32(var1);
	fprintf(stdout, "output: %d\n", output);
	return 0;
}

// from: https://github.com/brianrhall/Assembly/blob/master/Chapter_9/Program%209.2/Program_9.2b_VisualC.cpp
union S64d {
	__m128d v;
	double_t dubs[2];
};

int test_inline_float()
{
	S64d PIES;

	PIES.v = _mm_load1_pd(&PI);
	PIES.v = _mm_sqrt_sd(PIES.v, PIES.v);

	fprintf(stdout, "PIES.dubs[0]: %f\n", PIES.dubs[0]);
	return 0;
}
