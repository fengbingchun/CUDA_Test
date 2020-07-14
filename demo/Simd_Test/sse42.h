#include <mmintrin.h> //MMX
#include <xmmintrin.h> //SSE(include mmintrin.h)
#include <emmintrin.h> //SSE2(include xmmintrin.h)
#include <pmmintrin.h> //SSE3(include emmintrin.h)
#include <tmmintrin.h>//SSSE3(include pmmintrin.h)
#include <smmintrin.h>//SSE4.1(include tmmintrin.h)
#include <nmmintrin.h>//SSE4.2(include smmintrin.h)
#include <wmmintrin.h>//AES(include nmmintrin.h)
#include <immintrin.h>//AVX(include wmmintrin.h)
#include <intrin.h>//(include immintrin.h)

void SSE42FUN()
{
	/*
	* Intrinsics for text/string processing.
	*/
	//Either the computed mask of MaxSize bits or its expansion to a 128-bit parameter.
	//If the return value is expanded, each bit of the result mask is expanded to a 
	//byte or a word.详见参考文献2
	extern __m128i _mm_cmpistrm (__m128i a, __m128i b, const int mode);
	//An integer between 0 and Maxsize. MaxSize when the computed mask equals 0.
	//Otherwise, the index of the leftmost or rightmost bit set to 1 in this mask.
	//详见参考文献2
	extern int     _mm_cmpistri (__m128i a, __m128i b, const int mode);
	//Either the computed mask of MaxSize bits or its expansion to a 128-bit parameter.
	//If the return value is expanded, each bit of the result mask is expanded to 
	//a byte or a word.详见参考文献3
	extern __m128i _mm_cmpestrm (__m128i a, int la, __m128i b, int lb, const int mode);
	//An integer that ranges between 0 and MaxSize. Maxsize is returned when the 
	//resulting bitmask is equal to 0. Otherwise, the index of either the leftmost
	//or rightmost bit set to 1 in this mask.详见参考文献3
	extern int     _mm_cmpestri (__m128i a, int la, __m128i b, int lb, const int mode);

	/*
	* Intrinsics for text/string processing and reading values of EFlags.
	*/
	//Returns one if the null character occurs in b. Otherwise, zero. When one is 
	//returned, it means that b contains the ending fragment of the string that is 
	//being compared.详见参考文献2
	extern int     _mm_cmpistrz (__m128i a, __m128i b, const int mode);
	//Zero if the resulting mask is equal to zero. Otherwise, one.
	//详见参考文献2
	extern int     _mm_cmpistrc (__m128i a, __m128i b, const int mode);
	//One if the null character occurs in a. Otherwise, zero. When one is returned,
	//it means that a contains the ending fragment of the string that is being compared.
	//详见参考文献2
	extern int     _mm_cmpistrs (__m128i a, __m128i b, const int mode);
	//bit0 of the resulting bitmask.详见参考文献2
	extern int     _mm_cmpistro (__m128i a, __m128i b, const int mode);
	//One if b is does not contain the null character and the resulting mask is 
	//equal to zero. Otherwise, zero. 详见参考文献2
	extern int     _mm_cmpistra (__m128i a, __m128i b, const int mode);
	//One if the absolute value of lb is less than MaxSize. Otherwise, zero.详见参考文献3
	extern int     _mm_cmpestrz (__m128i a, int la, __m128i b, int lb, const int mode);
	//Zero if the resulting mask is equal to zero. Otherwise, one.详见参考文献3
	extern int     _mm_cmpestrc (__m128i a, int la, __m128i b, int lb, const int mode);
	//One if the absolute value of la is less than MaxSize. Otherwise, zero.详见参考文献3
	extern int     _mm_cmpestrs (__m128i a, int la, __m128i b, int lb, const int mode);
	//bit0 of the resulting bitmask. 详见参考文献3
	extern int     _mm_cmpestro (__m128i a, int la, __m128i b, int lb, const int mode);
	//One if the absolute value of lb is larger than or equal to MaxSize and the 
	//resulting mask is equal to zero. Otherwise, zero.详见参考文献3
	extern int     _mm_cmpestra (__m128i a, int la, __m128i b, int lb, const int mode);

	/*
	* Packed integer 64-bit comparison, zeroing or filling with ones
	* corresponding parts of result
	*/
	//val1=(val10, val11), val2=(val20, val21)
	//则,r0 = (val10 > val20) ? 0xffffffffffffffff : 0x0
	//	 r1 = (val11 > val21) ? 0xffffffffffffffff : 0x0
	extern __m128i _mm_cmpgt_epi64(__m128i val1, __m128i val2);

	/*
	* Calculate a number of bits set to 1
	*/
	//The number of bits set to one in v
	extern int _mm_popcnt_u32(unsigned int v);
	//The number of bits set to one in v
	extern __int64 _mm_popcnt_u64(unsigned __int64 v);

	/*
	* Accumulate CRC32 (polynomial 0x11EDC6F41) value
	*/
	//crc：循环冗余校验码,CRC32-C algorithm is based on polynomial 0x1EDC6F41,
	//r = crc + CRC-32C(v)
	extern unsigned int _mm_crc32_u8 (unsigned int crc, unsigned char v);
	//crc：循环冗余校验码,CRC32-C algorithm is based on polynomial 0x1EDC6F41,
	//r = crc + CRC-32C(v)
	extern unsigned int _mm_crc32_u16(unsigned int crc, unsigned short v);
	//crc：循环冗余校验码,CRC32-C algorithm is based on polynomial 0x1EDC6F41,
	//r = crc + CRC-32C(v)
	extern unsigned int _mm_crc32_u32(unsigned int crc, unsigned int v);
	//crc：循环冗余校验码,CRC32-C algorithm is based on polynomial 0x1EDC6F41,
	//r = crc + CRC-32C(v)
	extern unsigned __int64 _mm_crc32_u64(unsigned __int64 crc, unsigned __int64 v);
}
