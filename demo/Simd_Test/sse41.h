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

void SSE41FUN()
{
	/*Integer blend instructions - select data from 2 sources
	using constant/variable mask*/
	//v1=(v10, v11, ..., v17), v2=(v20, v21, ..., v27)
	//mask:If the corresponding flag bit is 0, the value is selected from parameter v1.
	//Otherwise the value is from parameter v2.
	//则r0=(mask0 == 0) ? v10 : v20,...,r7= (mask7 == 0) ? v17 : v27
	extern __m128i _mm_blend_epi16 (__m128i v1, __m128i v2, const int mask);
	//v1=(v10, v11, ..., v115), v2=(v20, v21, ..., v215), mask=(mask1, ..., mask15)
	//则r0=(mask0 & 0x80) ? v20 : v10, ..., r15=(mask15 & 0x80) ? v215 : v115
	extern __m128i _mm_blendv_epi8 (__m128i v1, __m128i v2, __m128i mask);

	/*Float single precision blend instructions - select data
	from 2 sources using constant/variable mask */
	//v1=(v10, v11, v12, v13), v2=(v20, v21, v22, v23)
	//则r0=(mask0 == 0) ? v10 : v20,..., r3= (mask3 == 0) ? v13 : v23
	extern __m128  _mm_blend_ps (__m128  v1, __m128  v2, const int mask);
	//v1=(v10, v11, v12, v13), v2=(v20, v21, v22, v23)
	//则r0= (v30 & 0x80000000) ? v20 : v10,...,r3= (v33 & 0x80000000) ? v23 : v13
	extern __m128  _mm_blendv_ps(__m128  v1, __m128  v2, __m128 v3);

	/*Float double precision blend instructions - select data
	from 2 sources using constant/variable mask*/
	//v1=(v10, v11), v2=(v20, v21)
	//则r0 = (mask0 == 0) ? v10 : v20, r1 = (mask1 == 0) ? v11 : v21
	extern __m128d _mm_blend_pd (__m128d v1, __m128d v2, const int mask);
	//v1=(v10, v11), v2=(v20, v21)
	//则r0 = (v30 & 0x8000000000000000) ? v20 : v10,
	//r1 = (v31 & 0x8000000000000000) ? v21 : v11
	extern __m128d _mm_blendv_pd(__m128d v1, __m128d v2, __m128d v3);

	/*Dot product instructions with mask-defined summing and zeroing
	of result's parts*/
	//val1=(val10, ..., val13), val2=(val20,...,val23)
	/*则tmp0 := (mask4 == 1) ? (val10 * val20) : +0.0
		tmp1 := (mask5 == 1) ? (val11 * val21) : +0.0
		tmp2 := (mask6 == 1) ? (val12 * val22) : +0.0
		tmp3 := (mask7 == 1) ? (val13 * val23) : +0.0
		tmp4 := tmp0 + tmp1 + tmp2 + tmp3
		r0 := (mask0 == 1) ? tmp4 : +0.0
		r1 := (mask1 == 1) ? tmp4 : +0.0
		r2 := (mask2 == 1) ? tmp4 : +0.0
		r3 := (mask3 == 1) ? tmp4 : +0.0 */
	extern __m128  _mm_dp_ps(__m128  val1, __m128  val2, const int mask);
	//val1=(val10, val11), val2=(val20, val21)
	/*则tmp0 := (mask4 == 1) ? (val10 * val20) : +0.0
		tmp1 := (mask5 == 1) ? (val11 * val21) : +0.0
		tmp2 := tmp0 + tmp1
		r0 := (mask0 == 1) ? tmp2 : +0.0
		r1 := (mask1 == 1) ? tmp2 : +0.0 */
	extern __m128d _mm_dp_pd(__m128d val1, __m128d val2, const int mask);

	/*Packed integer 64-bit comparison, zeroing or filling with ones
	corresponding parts of result */
	//val1=(val10, val11), val2=(val20, val21)
	//则r0 = (val10 == val20) ? 0xffffffffffffffff : 0,
	//r1 = (val11 == val21) ? 0xffffffffffffffff : 0
	extern __m128i _mm_cmpeq_epi64(__m128i val1, __m128i val2);

	/* Min/max packed integer instructions*/
	//val1=(val10,...,val115), val2=(val20,...,val215)
	//则r0 = (val10 < val20) ? val10 : val20, ...,
	//r15 = (val115 < val215) ? val115 : val215
	extern __m128i _mm_min_epi8 (__m128i val1, __m128i val2);
	//val1=(val10,...,val115), val2=(val20,...,val215)
	//则r0 = (val10 > val20) ? val10 : val20, ...,
	//r15 = (val115 > val215) ? val115 : val215
	extern __m128i _mm_max_epi8 (__m128i val1, __m128i val2);
	//val1=(val10,...,val17), val2=(val20,...,val27), eight 16-bit unsigned integers
	//则r0 = (val10 < val20) ? val10 : val20, ...,
	//r7 = (val17 < val27) ? val17 : val27
	extern __m128i _mm_min_epu16(__m128i val1, __m128i val2);
	//val1=(val10,...,val17), val2=(val20,...,val27),eight 16-bit unsigned integers
	//则r0 = (val10 > val20) ? val10 : val20, ...,
	//r7 = (val17 > val27) ? val17 : val27
	extern __m128i _mm_max_epu16(__m128i val1, __m128i val2);
	//val1=(val10,...,val13), val2=(val20,...,val23)
	//则r0 = (val10 < val20) ? val10 : val20, ...,
	//r3 = (val13 < val23) ? val13 : val23
	extern __m128i _mm_min_epi32(__m128i val1, __m128i val2);
	//val1=(val10,...,val13), val2=(val20,...,val23)
	//则r0 = (val10 > val20) ? val10 : val20, ...,
	//r3 = (val13 > val23) ? val13 : val23
	extern __m128i _mm_max_epi32(__m128i val1, __m128i val2);
	//val1=(val10,...,val13), val2=(val20,...,val23), four 32-bit unsigned integers
	//则r0 = (val10 < val20) ? val10 : val20, ...,
	//r3 = (val13 < val23) ? val13 : val23
	extern __m128i _mm_min_epu32(__m128i val1, __m128i val2);
	//val1=(val10,...,val13), val2=(val20,...,val23), four 32-bit unsigned integers
	//则r0 = (val10 > val20) ? val10 : val20, ...,
	//r3 = (val13 > val23) ? val13 : val23
	extern __m128i _mm_max_epu32(__m128i val1, __m128i val2);

	/*Packed integer 32-bit multiplication with truncation
	of upper halves of results*/
	//a=(a0,...,a3), b=(b0,...,b3), 则r0=a0 * b0, ..., r3=a3 * b3
	//Only the lower 32-bits of each product are saved
	extern __m128i _mm_mullo_epi32(__m128i a, __m128i b);

	/*Packed integer 32-bit multiplication of 2 pairs of operands
	producing two 64-bit results */
	//a=(a0,a1,a2,a3), b=(b0,b1,b2,b3)
	//r0=low_half(a0*b0), r1=high_half(a0*b0),r2=low_half(a2*b2), r3=high_half(a2*b2)
	//The upper 32-bits of each quadword of the input parameters are not used
	extern __m128i _mm_mul_epi32(__m128i a, __m128i b);

	/*Packed integer 128-bit bitwise comparison.
	return 1 if (val 'and' mask) == 0*/
	//则r = (mask & val) == 0, Generates a return value of 0 or 1
	extern int _mm_testz_si128(__m128i mask, __m128i val);

	/*Packed integer 128-bit bitwise comparison.
	return 1 if (val 'and_not' mask) == 0 */
	//则r=1 if all the bits set in val are set in mask; otherwise 0
	//Generates a return value of 0 or 1
	extern int _mm_testc_si128(__m128i mask, __m128i val);

	/*Packed integer 128-bit bitwise comparison
	ZF = ((val 'and' mask) == 0)  CF = ((val 'and_not' mask) == 0)
	return 1 if both ZF and CF are 0 */
	//则 ZF := (mask & s2) == 0,CF := (~mask & s2) == 0, r = ~ZF & ~CF
	//Generates a return value of 0 or 1
	extern int _mm_testnzc_si128(__m128i mask, __m128i s2);

	/*Insert single precision float into packed single precision
	array element selected by index.
	The bits [7-6] of the 3d parameter define src index,
	the bits [5-4] define dst index, and bits [3-0] define zeroing
	mask for dst */
	/*	sx := ndx6-7
		sval := (sx == 0) ? src0 : ((sx == 1) ? src1 : ((sx == 2) ? src2 : src3))

		dx := ndx4-5
		r0 := (dx == 0) ? sval : dst0
		r1 := (dx == 1) ? sval : dst1
		r2 := (dx == 2) ? sval : dst2
		r3 := (dx == 3) ? sval : dst3

		zmask := ndx0-3
		r0 := (zmask0 == 1) ? +0.0 : r0
		r1 := (zmask1 == 1) ? +0.0 : r1
		r2 := (zmask2 == 1) ? +0.0 : r2
		r3 := (zmask3 == 1) ? +0.0 : r3 */
	extern __m128 _mm_insert_ps(__m128 dst, __m128 src, const int ndx);

	/*Extract binary representation of single precision float from
	packed single precision array element selected by index */
	//src=(src0, src1, src2, src3)
	//则r = (ndx == 0) ? src0 : ((ndx == 1) ? src1 : ((ndx == 2) ? src2 : src3))
	//Only the least significant two bits of ndx are used
	extern int _mm_extract_ps(__m128 src, const int ndx);

	/*Insert integer into packed integer array element
	selected by index */
	//则r0=(ndx == 0) ? s : dst0, ..., r15=(ndx == 15) ? s : dst15
	//Only the lowest 8 bits of s are used, 
	//Only the least significant 4 bits of ndx are used
	extern __m128i _mm_insert_epi8 (__m128i dst, int s, const int ndx);
	//则r0=(ndx == 0) ? s : dst0, ..., r3=(ndx == 3) ? s : dst3
	//Only the least significant 2 bits of ndx are interpreted
	extern __m128i _mm_insert_epi32(__m128i dst, int s, const int ndx);
	//则r0=(ndx == 0) ? s : dst0, r1=(ndx == 1) ? s : dst1
	//Only the least significant bit of ndx is interpreted
	extern __m128i _mm_insert_epi64(__m128i dst, __int64 s, const int ndx);

	/*Extract integer from packed integer array element
	selected by index */
	//则r=(ndx == 0) ? src0 : ((ndx == 1) ? src1 : ...((ndx == 14) ? src14 : src15))
	//Only the least significant four bits of ndx are used
	//注意：The result is the unsigned equivalent of the appropriate 8-bits in parameter src
	extern int _mm_extract_epi8 (__m128i src, const int ndx);
	//则r=(ndx == 0) ? src0 : ((ndx == 1) ? src1 : ((ndx == 2) ? src2 : src3))
	//Only the least significant two bits of ndx are used.
	extern int _mm_extract_epi32(__m128i src, const int ndx);
	//则r = (ndx == 0) ? src0 : src1
	//Only the least significant bit of parameter ndx is used
	extern __int64 _mm_extract_epi64(__m128i src, const int ndx);

	/*Horizontal packed word minimum and its index in
	result[15:0] and result[18:16] respectively */
	//The lowest order 16 bits are the minimum value found in parameter shortValues.
	//The second-lowest order 16 bits are the index of the minimum value 
	//found in parameter shortValues.
	extern __m128i _mm_minpos_epu16(__m128i shortValues);

	/* Packed/single float double precision rounding */
	//则r0=RND(val0), r1=RND(val1),详见参考文献1
	extern __m128d _mm_round_pd(__m128d val, int iRoundMode);
	//则r0=RND(val0), r1=dst1, 详见参考文献1
	// The lowest 64 bits are the result of the rounding function on val.
	//The higher order 64 bits are copied directly from input parameter dst
	extern __m128d _mm_round_sd(__m128d dst, __m128d val, int iRoundMode);

	/*Packed/single float single precision rounding */
	//则r0=RND(val0), r1=RND(val1), r2=RND(val2), r3=RND(val3),详见参考文献1
	extern __m128  _mm_round_ps(__m128  val, int iRoundMode);
	//则r0=RND(val0), r1=dst1, r2=dst2, r3=dst3, 	
	//The lowest 32 bits are the result of the rounding function on val.
	//The higher order 96 bits are copied directly from input parameter dst
	extern __m128  _mm_round_ss(__m128 dst, __m128  val, int iRoundMode);

	/*Packed integer sign-extension */
	//byteValues: A 128-bit parameter that contains four signed 8-bit integers
	//in the lower 32 bits, byteValues=(a0, a1, ..., a15)
	/*则r0 := a0
		r1 := (a0 < 0) ? 0xff : 0
		r2 := (a0 < 0) ? 0xff : 0
		r3 := (a0 < 0) ? 0xff : 0

		r4 := a1
		r5 := (a1 < 0) ? 0xff : 0
		r6 := (a1 < 0) ? 0xff : 0
		r7 := (a1 < 0) ? 0xff : 0

		r8 := a2
		r9 := (a2 < 0) ? 0xff : 0
		r10 := (a2 < 0) ? 0xff : 0
		r11 := (a2 < 0) ? 0xff : 0

		r12 := a3
		r13 := (a3 < 0) ? 0xff : 0
		r14 := (a3 < 0) ? 0xff : 0
		r15 := (a3 < 0) ? 0xff : 0 */
	extern __m128i _mm_cvtepi8_epi32 (__m128i byteValues);
	//shortValues: A 128-bit parameter that contains four signed 16-bit integers
	//in the lower 64 bits, shortValues=(a0, a1, ..., a7)
	/*则r0 := a0
		r1 := (a0 < 0) ? 0xffff : 0

		r2 := a1
		r3 := (a1 < 0) ? 0xffff : 0

		r4 := a2
		r5 := (a2 < 0) ? 0xffff : 0

		r6 := a3
		r7 := (a3 < 0) ? 0xffff : 0 */
	extern __m128i _mm_cvtepi16_epi32(__m128i shortValues);
	//byteValues: A 128-bit parameter that contains two signed 8-bit integers
	//in the lower 16 bits, byteValues=(a0, a1, ... , a15)
	/*则r0 := a0
		r1 := (a0 < 0) ? 0xff : 0
		r2 := (a0 < 0) ? 0xff : 0
		r3 := (a0 < 0) ? 0xff : 0
		r4 := (a0 < 0) ? 0xff : 0
		r5 := (a0 < 0) ? 0xff : 0
		r6 := (a0 < 0) ? 0xff : 0
		r7 := (a0 < 0) ? 0xff : 0

		r8 := a1
		r9 := (a1 < 0) ? 0xff : 0
		r10 := (a1 < 0) ? 0xff : 0
		r11 := (a1 < 0) ? 0xff : 0
		r12 := (a1 < 0) ? 0xff : 0
		r13 := (a1 < 0) ? 0xff : 0
		r14 := (a1 < 0) ? 0xff : 0
		r15 := (a1 < 0) ? 0xff : 0 */
	extern __m128i _mm_cvtepi8_epi64 (__m128i byteValues); 
	//intValues: A 128-bit parameter that contains two signed 32-bit 
	//integers in the lower 64 bits, intValues=(a0, a1, a2, a3)
	/*则r0 := a0
		r1 := (a0 < 0) ? 0xffffffff : 0
		r2 := a1
		r3 := (a1 < 0) ? 0xffffffff : 0*/
	extern __m128i _mm_cvtepi32_epi64(__m128i intValues);
	//shortValues:A 128-bit parameter that contains two signed 16-bit integers
	//in the lower 32 bits, shortValues=(a0, a1, ..., a7)
	/*则r0 := a0
		r1 := (a0 < 0) ? 0xffff : 0
		r2 := (a0 < 0) ? 0xffff : 0
		r3 := (a0 < 0) ? 0xffff : 0

		r4 := a1
		r5 := (a1 < 0) ? 0xffff : 0
		r6 := (a1 < 0) ? 0xffff : 0
		r7 := (a1 < 0) ? 0xffff : 0*/
	extern __m128i _mm_cvtepi16_epi64(__m128i shortValues);
	//byteValues:A 128-bit parameter that contains eight signed 8-bit integers 
	//in the lower 64 bits, byteValues=(a0, a1, ..., a15)
	/*则r0 := a0
		r1 := (a0 < 0) ? 0xff : 0
		r2 := a1
		r3 := (a1 < 0) ? 0xff : 0
		...
		r14 := a7
		r15 := (a7 < 0) ? 0xff : 0*/
	extern __m128i _mm_cvtepi8_epi16 (__m128i byteValues);

	/*Packed integer zero-extension*/
	//byteValues:A 128-bit parameter that contains four unsigned 8-bit integers
	//in the lower 32 bits, byteValues=(a0, a1, ... , a15)
	/*则r0 := a0
		r1 := 0
		r2 := 0
		r3 := 0

		r4 := a1
		r5 := 0
		r6 := 0
		r7 := 0

		r8 := a2
		r9 := 0
		r10 := 0
		r11 := 0

		r12 := a3
		r13 := 0
		r14 := 0
		r15 := 0*/
	extern __m128i _mm_cvtepu8_epi32 (__m128i byteValues);
	//shortValues:A 128-bit parameter that contains four unsigned 16-bit integers
	//in the lower 64 bits, shortValues=(a0, a1, ... , a7)
	/*则r0 := a0
		r1 := 0

		r2 := a1
		r3 := 0

		r4 := a2
		r5 := 0

		r6 := a3
		r7 := 0*/
	extern __m128i _mm_cvtepu16_epi32(__m128i shortValues);
	//shortValues:A 128-bit parameter that contains two unsigned 8-bit integers
	//in the lower 16 bits, shortValues=(a0, a1, ..., a15)
	/*则r0 := a0
		r1 := 0
		r2 := 0
		r3 := 0
		r4 := 0
		r5 := 0
		r6 := 0
		r7 := 0

		r8 := a1
		r9 := 0
		r10 := 0
		r11 := 0
		r12 := 0
		r13 := 0
		r14 := 0
		r15 := 0*/
	extern __m128i _mm_cvtepu8_epi64 (__m128i shortValues);
	//intValues:A 128-bit parameter that contains two unsigned 32-bit integers
	//in the lower 64 bits, intValues=(a0, a1, a2, a3)
	/*则r0 = a0
		r1 = 0
		r2 = a1
		r3 = 0*/
	extern __m128i _mm_cvtepu32_epi64(__m128i intValues);
	//shortValues:A 128-bit parameter that contains two unsigned 16-bit integers
	//in the lower 32 bits, shortValues=(a0, a1, ... , a7)
	/*则r0 := a0
		r1 := 0
		r2 := 0
		r3 := 0

		r4 := a1
		r5 := 0
		r6 := 0
		r7 := 0*/
	extern __m128i _mm_cvtepu16_epi64(__m128i shortValues);
	//byteValues:A 128-bit parameter that contains eight unsigned 8-bit integers 
	//in the lower 64 bits, byteValues=(a0, a1, ... , a15)
	/*则r0 := a0
		r1 := 0
		r2 := a1
		r3 := 0
		...
		r14 := a7
		r15 := 0*/
	extern __m128i _mm_cvtepu8_epi16 (__m128i byteValues);

	/*Pack 8 double words from 2 operands into 8 words of result
	with unsigned saturation */
	//val1=(val10,...,vall3), val2=(val20, ..., val23)
	/*则r0 := (val10 < 0) ? 0 : ((val10 > 0xffff) ? 0xffff : val10)
		r1 := (val11 < 0) ? 0 : ((val11 > 0xffff) ? 0xffff : val11)
		r2 := (val12 < 0) ? 0 : ((val12 > 0xffff) ? 0xffff : val12)
		r3 := (val13 < 0) ? 0 : ((val13 > 0xffff) ? 0xffff : val13)
		r4 := (val20 < 0) ? 0 : ((val20 > 0xffff) ? 0xffff : val20)
		r5 := (val21 < 0) ? 0 : ((val21 > 0xffff) ? 0xffff : val21)
		r6 := (val22 < 0) ? 0 : ((val22 > 0xffff) ? 0xffff : val22)
		r7 := (val23 < 0) ? 0 : ((val23 > 0xffff) ? 0xffff : val23)*/
	extern __m128i _mm_packus_epi32(__m128i val1, __m128i val2);

	/*Sum absolute 8-bit integer difference of adjacent groups of 4 byte
	integers in operands. Starting offsets within operands are
	determined by mask */
	//s1, s2: sixteen 8-bit unsigned integers
	// msk0, msk1, and msk2 are the three least significant bits of parameter msk
	/*则i = msk2 * 4
		j = msk0-1 * 4
		for (k = 0; k < 8; k = k + 1) {
		t0 = abs(s1[i + k + 0] - s2[j + 0])
		t1 = abs(s1[i + k + 1] - s2[j + 1])
		t2 = abs(s1[i + k + 2] - s2[j + 2])
		t3 = abs(s1[i + k + 3] - s2[j + 3])
		r[k] = t0 + t1 + t2 + t3
		}*/
	extern __m128i _mm_mpsadbw_epu8(__m128i s1, __m128i s2, const int msk);

	/*
	* Load double quadword using non-temporal aligned hint
	*/
	//This instruction loads data from a specified address.The memory source must be 
	//16-byte aligned because the return value consists of sixteen bytes.则r=*v1
	extern __m128i _mm_stream_load_si128(__m128i* v1);
}
