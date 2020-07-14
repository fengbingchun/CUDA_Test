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

void SSSE3FUN()
{
	/*Add horizonally packed [saturated] words, double words,
	{X,}MM2/m{128,64} (b) to {X,}MM1 (a).*/
	//a=(a0, a1, a2, a3, a4, a5, a6, a7), b=(b0, b1, b2, b3, b4, b5, b6, b7)
	//则r0=a0+a1, r1=a2+a3, r2=a4+a5, r3=a6+a7, r4=b0+b1, r5=b2+b3, r6=b4+b5, r7=b6+b7 
	extern __m128i _mm_hadd_epi16 (__m128i a, __m128i b);
	//a=(a0, a1, a2, a3), b=(b0, b1, b2, b3)
	//则r0=a0+a1, r1=a2+a3, r2=b0+b1, r3=b2+b3
	extern __m128i _mm_hadd_epi32 (__m128i a, __m128i b);
	//SATURATE_16(x) is ((x > 32767) ? 32767 : ((x < -32768) ? -32768 : x))
	//a=(a0, a1, a2, a3, a4, a5, a6, a7), b=(b0, b1, b2, b3, b4, b5, b6, b7)
	//则r0=SATURATE_16(a0+a1), ..., r3=SATURATE_16(a6+a7), 
	//r4=SATURATE_16(b0+b1), ..., r7=SATURATE_16(b6+b7)
	extern __m128i _mm_hadds_epi16 (__m128i a, __m128i b);
	//a=(a0, a1, a2, a3), b=(b0, b1, b2, b3)
	//则r0=a0+a1, r1=a2+a3, r2=b0+b1, r3=b2+b3
	extern __m64 _mm_hadd_pi16 (__m64 a, __m64 b);
	//a=(a0, a1), b=(b0, b1), 则r0=a0+a1, r1=b0+b1
	extern __m64 _mm_hadd_pi32 (__m64 a, __m64 b);
	//SATURATE_16(x) is ((x > 32767) ? 32767 : ((x < -32768) ? -32768 : x))
	//a=(a0, a1, a2, a3), b=(b0, b1, b2, b3)
	//则r0=SATURATE_16(a0+a1), r1=SATURATE_16(a2+a3), 
	//r2=SATURATE_16(b0+b1), r3=SATURATE_16(b2+b3)
	extern __m64 _mm_hadds_pi16 (__m64 a, __m64 b);

	/*Subtract horizonally packed [saturated] words, double words,
	{X,}MM2/m{128,64} (b) from {X,}MM1 (a).*/
	//a=(a0, a1, a2, a3, a4, a5, a6, a7), b=(b0, b1, b2, b3, b4, b5, b6, b7)
	//则r0=a0-a1, r1=a2-a3, r2=a4-a5, r3=a6-a7, r4=b0-b1, r5=b2-b3, r6=b4-b5, r7=b6-b7
	extern __m128i _mm_hsub_epi16 (__m128i a, __m128i b);
	//a=(a0, a1, a2, a3), b=(b0, b1, b2, b3)
	//则r0=a0-a1, r1=a2-a3, r2=b0-b1, r3=b2-b3
	extern __m128i _mm_hsub_epi32 (__m128i a, __m128i b);
	//SATURATE_16(x) is ((x > 32767) ? 32767 : ((x < -32768) ? -32768 : x))
	//a=(a0, a1, a2, a3, a4, a5, a6, a7), b=(b0, b1, b2, b3, b4, b5, b6, b7)
	//则r0=SATURATE_16(a0-a1), ..., r3=SATURATE_16(a6-a7), 
	//r4=SATURATE_16(b0-b1), ..., r7=SATURATE_16(b6-b7)
	extern __m128i _mm_hsubs_epi16 (__m128i a, __m128i b);
	//a=(a0, a1, a2, a3), b=(b0, b1, b2, b3)
	//则r0=a0-a1, r1=a2-a3, r2=b0-b1, r3=b2-b3
	extern __m64 _mm_hsub_pi16 (__m64 a, __m64 b);
	//a=(a0, a1), b=(b0, b1), 则r0=a0-a1, r1=b0-b1
	extern __m64 _mm_hsub_pi32 (__m64 a, __m64 b);
	//SATURATE_16(x) is ((x > 32767) ? 32767 : ((x < -32768) ? -32768 : x))
	//a=(a0, a1, a2, a3), b=(b0, b1, b2, b3)
	//则r0=SATURATE_16(a0-a1), r1=SATURATE_16(a2-a3), 
	//r2=SATURATE_16(b0-b1), r3=SATURATE_16(b2-b3)
	extern __m64 _mm_hsubs_pi16 (__m64 a, __m64 b);

	/*Multiply and add packed words,
	{X,}MM2/m{128,64} (b) to {X,}MM1 (a).*/
	//SATURATE_16(x) is ((x > 32767) ? 32767 : ((x < -32768) ? -32768 : x))
	//a=(a0, a1, a2, ..., a13, a14, a15), b=(b0, b1, b2, ..., b13, b14, b15)
	//则r0=SATURATE_16((a0*b0)+(a1*b1)), ..., r7=SATURATE_16((a14*b14)+(a15*b15))
	//Parameter a contains unsigned bytes. Parameter b contains signed bytes.
	extern __m128i _mm_maddubs_epi16 (__m128i a, __m128i b);
	//SATURATE_16(x) is ((x > 32767) ? 32767 : ((x < -32768) ? -32768 : x))
	//a=(a0, a1, a2, a3, a4, a5, a6, a7), b=(b0, b1, b2, b3, b4, b5, b6, b7)
	//则r0=SATURATE_16((a0*b0)+(a1*b1)), ..., r3=SATURATE_16((a6*b6)+(a7*b7))
	//Parameter a contains unsigned bytes. Parameter b contains signed bytes.
	extern __m64 _mm_maddubs_pi16 (__m64 a, __m64 b);

	/*Packed multiply high integers with round and scaling,
	{X,}MM2/m{128,64} (b) to {X,}MM1 (a).*/
	//a=(a0, a1, a2, a3, a4, a5, a6, a7), b=(b0, b1, b2, b3, b4, b5, b6, b7)
	//则r0=INT16(((a0*b0)+0x4000) >> 15), ..., r7=INT16(((a7*b7)+0x4000) >> 15)
	extern __m128i _mm_mulhrs_epi16 (__m128i a, __m128i b);
	//a=(a0, a1, a2, a3), b=(b0, b1, b2, b3)
	//则r0=INT16(((a0*b0)+0x4000) >> 15), ..., r3=INT16(((a3*b3)+0x4000) >> 15)
	extern __m64 _mm_mulhrs_pi16 (__m64 a, __m64 b);

	/*Packed shuffle bytes
	{X,}MM2/m{128,64} (b) by {X,}MM1 (a).*/
	//SELECT(a, n) extracts the nth 8-bit parameter from a. The 0th 8-bit parameter
	//is the least significant 8-bits, b=(b0, b1, b2, ..., b13, b14, b15), b is mask
	//则r0 = (b0 & 0x80) ? 0 : SELECT(a, b0 & 0x0f), ...,
	//r15 = (b15 & 0x80) ? 0 : SELECT(a, b15 & 0x0f)
	extern __m128i _mm_shuffle_epi8 (__m128i a, __m128i b);
	//SELECT(a, n) extracts the nth 8-bit parameter from a. The 0th 8-bit parameter
	//is the least significant 8-bits, b=(b0, b1, ..., b7), b is mask
	//则r0= (b0 & 0x80) ? 0 : SELECT(a, b0 & 0x07),...,
	//r7=(b7 & 0x80) ? 0 : SELECT(a, b7 & 0x07)
	extern __m64 _mm_shuffle_pi8 (__m64 a, __m64 b);

	/*Packed byte, word, double word sign, {X,}MM2/m{128,64} (b) to
	{X,}MM1 (a).*/
	//a=(a0, a1, a2, ..., a13, a14, a15), b=(b0, b1, b2, ..., b13, b14, b15)
	//则r0=(b0 < 0) ? -a0 : ((b0 == 0) ? 0 : a0), ...,
	//r15= (b15 < 0) ? -a15 : ((b15 == 0) ? 0 : a15)
	extern __m128i _mm_sign_epi8 (__m128i a, __m128i b);
	//a=(a0, a1, a2, a3, a4, a5, a6, a7), b=(b0, b1, b2, b3, b4, b5, b6, b7)
	//r0=(b0 < 0) ? -a0 : ((b0 == 0) ? 0 : a0), ...,
	//r7= (b7 < 0) ? -a7 : ((b7 == 0) ? 0 : a7)
	extern __m128i _mm_sign_epi16 (__m128i a, __m128i b);
	//a=(a0, a1, a2, a3), b=(b0, b1, b2, b3)
	//则r0=(b0 < 0) ? -a0 : ((b0 == 0) ? 0 : a0), ...,
	//r3= (b3 < 0) ? -a3 : ((b3 == 0) ? 0 : a3)
	extern __m128i _mm_sign_epi32 (__m128i a, __m128i b);
	//a=(a0, a1, a2, a3, a4, a5, a6, a7), b=(b0, b1, b2, b3, b4, b5, b6, b7)
	//则r0=(b0 < 0) ? -a0 : ((b0 == 0) ? 0 : a0), ...,
	//r7= (b7 < 0) ? -a7 : ((b7 == 0) ? 0 : a7)
	extern __m64 _mm_sign_pi8 (__m64 a, __m64 b);
	//a=(a0, a1, a2, a3), b=(b0, b1, b2, b3)
	//则r0=(b0 < 0) ? -a0 : ((b0 == 0) ? 0 : a0), ...,
	//r3= (b3 < 0) ? -a3 : ((b3 == 0) ? 0 : a3)
	extern __m64 _mm_sign_pi16 (__m64 a, __m64 b);
	//a=(a0, a1), b=(b0, b1), 则r0=(b0 < 0) ? -a0 : ((b0 == 0) ? 0 : a0),
	//r1= (b1 < 0) ? -a1 : ((b1 == 0) ? 0 : a1)
	extern __m64 _mm_sign_pi32 (__m64 a, __m64 b);

	/*Packed align and shift right by n*8 bits,
	{X,}MM2/m{128,64} (b) to {X,}MM1 (a).*/
	//n: A constant that specifies how many bytes the interim result will be 
	//shifted to the right, If n > 32, the result value is zero 
	//CONCAT(a, b) is the 256-bit unsigned intermediate value that is a concatenation of 
	//parameters a and b. The result is this intermediate value shifted right by n bytes.
	//则r= (CONCAT(a, b) >> (n * 8)) & 0xffffffffffffffff
	extern __m128i _mm_alignr_epi8 (__m128i a, __m128i b, int n);
	//n: An integer constant that specifies how many bytes to shift the interim 
	//result to the right,If n > 16, the result value is zero
	//CONCAT(a, b) is the 128-bit unsigned intermediate value that is formed by 
	//concatenating parameters a and b. The result value is the rightmost 64 bits after
	//shifting this intermediate result right by n bytes
	//则r = (CONCAT(a, b) >> (n * 8)) & 0xffffffff
	extern __m64 _mm_alignr_pi8 (__m64 a, __m64 b, int n);

	/*Packed byte, word, double word absolute value,
	{X,}MM2/m{128,64} (b) to {X,}MM1 (a).*/
	//a=(a0, a1, a2, ..., a13, a14, a15)
	//则r0 = (a0 < 0) ? -a0 : a0, ..., r15 = (a15 < 0) ? -a15 : a15
	extern __m128i _mm_abs_epi8 (__m128i a);
	//a=(a0, a1, a2, a3, a4, a5, a6, a7)
	//则r0 = (a0 < 0) ? -a0 : a0, ..., r7 = (a7 < 0) ? -a7 : a7
	extern __m128i _mm_abs_epi16 (__m128i a);
	//a=(a0, a1, a2, a3)
	//则r0 = (a0 < 0) ? -a0 : a0, ..., r3 = (a3 < 0) ? -a3 : a3
	extern __m128i _mm_abs_epi32 (__m128i a);
	//a=(a0, a1, a2, a3, a4, a5, a6, a7)
	//则r0 = (a0 < 0) ? -a0 : a0, ..., r7 = (a7 < 0) ? -a7 : a7
	extern __m64 _mm_abs_pi8 (__m64 a);
	//a=(a0, a1, a2, a3)
	//则r0 = (a0 < 0) ? -a0 : a0, ..., r3 = (a3 < 0) ? -a3 : a3
	extern __m64 _mm_abs_pi16 (__m64 a);
	//a=(a0, a1), 则r0 = (a0 < 0) ? -a0 : a0, r1 = (a1 < 0) ? -a1 : a1
	extern __m64 _mm_abs_pi32 (__m64 a);
}
