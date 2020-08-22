#include "funset.hpp"
#include <iostream>
#include <algorithm>
#include <immintrin.h>

namespace {

void Integer_Add(const unsigned char* p1, const unsigned char* p2, unsigned char* p3, int num)
{
	__m128i m1 = _mm_loadu_si128((__m128i*)p1);
	__m128i m2 = _mm_loadu_si128((__m128i*)p2);
	__m128i m3 = _mm_add_epi8(m1, m2);

	_mm_storeu_si128((__m128i*)p3, m3);
}

void Integer_Sub(const unsigned char* p1, const unsigned char* p2, unsigned char* p3, int num)
{
	__m128i m1 = _mm_loadu_si128((__m128i*)p1);
	__m128i m2 = _mm_loadu_si128((__m128i*)p2);

	__m128i m3 = _mm_setzero_si128();
	m3 = _mm_subs_epi8(m1, m2);

	_mm_storeu_si128((__m128i*)p3, m3);
}

void Integer_Avg(const unsigned char* p1, const unsigned char* p2, unsigned char* p3, int num)
{
	__m128i m1 = _mm_loadu_si128((__m128i*)p1);
	__m128i m2 = _mm_loadu_si128((__m128i*)p2);

	__m128i m3 = _mm_setzero_si128();
	m3 = _mm_avg_epu8(m1, m2);

	_mm_storeu_si128((__m128i*)p3, m3);
}

void print(const unsigned char* data, int num)
{
	fprintf(stdout, "result: ");
	std::for_each(data, data+num, [](const unsigned char& n){ fprintf(stdout, "0x%02x, ", n); });
	fprintf(stdout, "\n");
}

} // namespace

int test_simd_1()
{
	const int num = 16;
	unsigned char array1[num] = { 0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0 };
	unsigned char array2[num] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f };
	unsigned char array3[num] = { 0 };

	Integer_Add(array1, array2, array3, num);
	print(&array3[0], num);

	Integer_Sub(array1, array2, array3, num);
	print(&array3[0], num);

	Integer_Avg(array1, array2, array3, num);
	print(&array3[0], num);

	return 0;
}

