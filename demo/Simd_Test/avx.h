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

void AVXFUN()
{
	/*
	* Add Packed Double Precision Floating-Point Values
	* **** VADDPD ymm1, ymm2, ymm3/m256
	* Performs an SIMD add of the four packed double-precision floating-point
	* values from the first source operand to the second source operand, and
	* stores the packed double-precision floating-point results in the
	* destination
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23)
	//则r0=m10+m20, r1=m11+m21, r2=m12+m22, r3=m13+m23
	extern __m256d __cdecl _mm256_add_pd(__m256d m1, __m256d m2);

	/*
	* Add Packed Single Precision Floating-Point Values
	* **** VADDPS ymm1, ymm2, ymm3/m256
	* Performs an SIMD add of the eight packed single-precision floating-point
	* values from the first source operand to the second source operand, and
	* stores the packed single-precision floating-point results in the
	* destination
	*/
	//m1=(m10, m11, ..., m17), m2=(m20, m21, ..., m27)
	//则r0=m10+m20, r1=m11+m21, ..., r7=m17+m27
	extern __m256 __cdecl _mm256_add_ps(__m256 m1, __m256 m2);

	/*
	* Add/Subtract Double Precision Floating-Point Values
	* **** VADDSUBPD ymm1, ymm2, ymm3/m256
	* Adds odd-numbered double-precision floating-point values of the first
	* source operand with the corresponding double-precision floating-point
	* values from the second source operand; stores the result in the odd-numbered
	* values of the destination. Subtracts the even-numbered double-precision
	* floating-point values from the second source operand from the corresponding
	* double-precision floating values in the first source operand; stores the
	* result into the even-numbered values of the destination
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23)
	//则r0=m10-m20, r1=m11+m21, r2=m12-m22, r3=m13-m23
	extern __m256d __cdecl _mm256_addsub_pd(__m256d m1, __m256d m2);

	/*
	* Add/Subtract Packed Single Precision Floating-Point Values
	* **** VADDSUBPS ymm1, ymm2, ymm3/m256
	* Adds odd-numbered single-precision floating-point values of the first source
	* operand with the corresponding single-precision floating-point values from
	* the second source operand; stores the result in the odd-numbered values of
	* the destination. Subtracts the even-numbered single-precision floating-point
	* values from the second source operand from the corresponding
	* single-precision floating values in the first source operand; stores the
	* result into the even-numbered values of the destination
	*/
	//m1=(m10, m11, m12, m13, ..., m17), m2=(m20, m21, m22, m23, ..., m27)
	//则r0=m10-m20, r1=m11+m21, ... , r6=m16-m26, r7=m17+m27
	extern __m256 __cdecl _mm256_addsub_ps(__m256 m1, __m256 m2);

	/*
	* Bitwise Logical AND of Packed Double Precision Floating-Point Values
	* **** VANDPD ymm1, ymm2, ymm3/m256
	* Performs a bitwise logical AND of the four packed double-precision
	* floating-point values from the first source operand and the second
	* source operand, and stores the result in the destination
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23)
	//则r0=(m10 & m20), r1=(m11 & m21), r2=(m12 & m22), r3=(m13 & m23)
	extern __m256d __cdecl _mm256_and_pd(__m256d m1, __m256d m2);

	/*
	* Bitwise Logical AND of Packed Single Precision Floating-Point Values
	* **** VANDPS ymm1, ymm2, ymm3/m256
	* Performs a bitwise logical AND of the eight packed single-precision
	* floating-point values from the first source operand and the second
	* source operand, and stores the result in the destination
	*/
	//m1=(m10, m11, m12, m13, ..., m17), m2=(m20, m21, m22, m23, ..., m27)
	//则r0=(m10 & m20), r1=(m11 & m21), ..., r6=(m16 & m26), r7=(m17 & m27)
	extern __m256 __cdecl _mm256_and_ps(__m256 m1, __m256 m2);

	/*
	* Bitwise Logical AND NOT of Packed Double Precision Floating-Point Values
	* **** VANDNPD ymm1, ymm2, ymm3/m256
	* Performs a bitwise logical AND NOT of the four packed double-precision
	* floating-point values from the first source operand and the second source
	* operand, and stores the result in the destination
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23)
	//则r0=(~m10) & m20, r1=(~m11) & m21, r2=(~m12) & m22, r3=(~m13) & m23
	extern __m256d __cdecl _mm256_andnot_pd(__m256d m1, __m256d m2);

	/*
	* Bitwise Logical AND NOT of Packed Single Precision Floating-Point Values
	* **** VANDNPS ymm1, ymm2, ymm3/m256
	* Performs a bitwise logical AND NOT of the eight packed single-precision
	* floating-point values from the first source operand and the second source
	* operand, and stores the result in the destination
	*/
	//m1=(m10, m11, m12, m13, ..., m17), m2=(m20, m21, m22, m23, ..., m27)
	//则r0=(~m10) & m20, r1=(~m11) & m21), ..., r6=(~m16) & m26, r7=(~m17) & m27
	extern __m256 __cdecl _mm256_andnot_ps(__m256 m1, __m256 m2);

	/*
	* Blend Packed Double Precision Floating-Point Values
	* **** VBLENDPD ymm1, ymm2, ymm3/m256, imm8
	* Double-Precision Floating-Point values from the second source operand are
	* conditionally merged with values from the first source operand and written
	* to the destination. The immediate bits [3:0] determine whether the
	* corresponding Double-Precision Floating Point value in the destination is
	* copied from the second source or first source. If a bit in the mask,
	* orresponding to a word, is "1", then the Double-Precision Floating-Point
	* value in the second source operand is copied, else the value in the first
	* source operand is copied
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23), mask=[b3 b2 b1 b0]
	//如果bn=1,则rn=m2n,如果bn=0, 则rn=m1n, 其中n为0,1,2,3
	extern __m256d __cdecl _mm256_blend_pd(__m256d m1, __m256d m2, const int mask);

	/*
	* Blend Packed Single Precision Floating-Point Values
	* **** VBLENDPS ymm1, ymm2, ymm3/m256, imm8
	* Single precision floating point values from the second source operand are
	* conditionally merged with values from the first source operand and written
	* to the destination. The immediate bits [7:0] determine whether the
	* corresponding single precision floating-point value in the destination is
	* copied from the second source or first source. If a bit in the mask,
	* corresponding to a word, is "1", then the single-precision floating-point
	* value in the second source operand is copied, else the value in the first
	* source operand is copied
	*/
	//m1=(m10, m11, ..., m17), m2=(m20, m21, ..., m27),mask=[b7 b6...b1 b0]
	//如果bn=1,则rn=m2n,如果bn=0, 则rn=m1n, 其中n为0,1,2,3,4,5,6,7
	extern __m256 __cdecl _mm256_blend_ps(__m256 m1, __m256 m2, const int mask);

	/*
	* Blend Packed Double Precision Floating-Point Values
	* **** VBLENDVPD ymm1, ymm2, ymm3/m256, ymm4
	* Conditionally copy each quadword data element of double-precision
	* floating-point value from the second source operand (third operand) and the
	* first source operand (second operand) depending on mask bits defined in the
	* mask register operand (fourth operand).
	*/
	extern __m256d __cdecl _mm256_blendv_pd(__m256d m1, __m256d m2, __m256d m3);

	/*
	* Blend Packed Single Precision Floating-Point Values
	* **** VBLENDVPS ymm1, ymm2, ymm3/m256, ymm4
	* Conditionally copy each dword data element of single-precision
	* floating-point value from the second source operand (third operand) and the
	* first source operand (second operand) depending on mask bits defined in the
	* mask register operand (fourth operand).
	*/
	extern __m256 __cdecl _mm256_blendv_ps(__m256 m1, __m256 m2, __m256 mask);

	/*
	* Divide Packed Double-Precision Floating-Point Values
	* **** VDIVPD ymm1, ymm2, ymm3/m256
	* Performs an SIMD divide of the four packed double-precision floating-point
	* values in the first source operand by the four packed double-precision
	* floating-point values in the second source operand
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23)
	//则r0=m10/m20, r1=m11/m21, r2=m12/m22, r3=m13/m23
	extern __m256d __cdecl _mm256_div_pd(__m256d m1, __m256d m2);

	/* 
	* Divide Packed Single-Precision Floating-Point Values
	* **** VDIVPS ymm1, ymm2, ymm3/m256
	* Performs an SIMD divide of the eight packed single-precision
	* floating-point values in the first source operand by the eight packed
	* single-precision floating-point values in the second source operand
	*/
	//m1=(m10, m11, m12, m13, ..., m17), m2=(m20, m21, m22, m23, ..., m27)
	//则r0=m10/m20, r1=m11/m21, ..., r6=m16/m26, r7=m17/m27
	extern __m256 __cdecl _mm256_div_ps(__m256 m1, __m256 m2);

	/*
	* Dot Product of Packed Single-Precision Floating-Point Values
	* **** VDPPS ymm1, ymm2, ymm3/m256, imm8
	* Multiplies the packed single precision floating point values in the
	* first source operand with the packed single-precision floats in the
	* second source. Each of the four resulting single-precision values is
	* conditionally summed depending on a mask extracted from the high 4 bits
	* of the immediate operand. This sum is broadcast to each of 4 positions
	* in the destination if the corresponding bit of the mask selected from
	* the low 4 bits of the immediate operand is "1". If the corresponding
	* low bit 0-3 of the mask is zero, the destination is set to zero.
	* The process is replicated for the high elements of the destination.
	*/
	//m1=(m10, m11, m12, m13, ..., m17), m2=(m20, m21, m22, m23, ..., m27)
	//mask=[b7 b6 ... b0], mask的低四位决定结果值是0,还是m1和m2相应位相乘后再求和
	//若b0b1b2b3为0001,则r0=r1=r2=0,m4=m5=m6=0,此时如果b4b5b6b7为1001,
	//则r3=m10*m20+m13*m23, r7=m14*m24+m17*m27,其它依次类推
	extern __m256 __cdecl _mm256_dp_ps(__m256 m1, __m256 m2, const int mask);

	/*
	* Add Horizontal Double Precision Floating-Point Values
	* **** VHADDPD ymm1, ymm2, ymm3/m256
	* Adds pairs of adjacent double-precision floating-point values in the
	* first source operand and second source operand and stores results in
	* the destination
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23)
	//则r0=m10+m11, r1=m20+m21, r2=m12+m13, r3=m22+m23
	extern __m256d __cdecl _mm256_hadd_pd(__m256d m1, __m256d m2);

	/*
	* Add Horizontal Single Precision Floating-Point Values
	* **** VHADDPS ymm1, ymm2, ymm3/m256
	* Adds pairs of adjacent single-precision floating-point values in the
	* first source operand and second source operand and stores results in
	* the destination
	*/
	//m1=(m10, m11, ..., m17), m2=(m20, m21, ..., m27)
	//则r0=m10+m11, r1=m12+m13, r2=m20+m21, r3=m22+m23,
	//r4=m14+m15, r5=m16+m17, r6=m24+m25, r7=m26+m27
	extern __m256 __cdecl _mm256_hadd_ps(__m256 m1, __m256 m2);

	/*
	* Subtract Horizontal Double Precision Floating-Point Values
	* **** VHSUBPD ymm1, ymm2, ymm3/m256
	* Subtract pairs of adjacent double-precision floating-point values in
	* the first source operand and second source operand and stores results
	* in the destination
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23)
	//则r0=m10-m11, r1=m20-m21, r2=m12-m13, r3=m22-m23
	extern __m256d __cdecl _mm256_hsub_pd(__m256d m1, __m256d m2);

	/*
	* Subtract Horizontal Single Precision Floating-Point Values
	* **** VHSUBPS ymm1, ymm2, ymm3/m256
	* Subtract pairs of adjacent single-precision floating-point values in
	* the first source operand and second source operand and stores results
	* in the destination.
	*/
	//m1=(m10, m11, ..., m17), m2=(m20, m21, ..., m27)
	//则r0=m10-m11, r1=m12-m13, r2=m20-m21, r3=m22-m23,
	//r4=m14-m15, r5=m16-m17, r6=m24-m25, r7=m26-m27
	extern __m256 __cdecl _mm256_hsub_ps(__m256 m1, __m256 m2);

	/*
	* Maximum of Packed Double Precision Floating-Point Values
	* **** VMAXPD ymm1, ymm2, ymm3/m256
	* Performs an SIMD compare of the packed double-precision floating-point
	* values in the first source operand and the second source operand and
	* returns the maximum value for each pair of values to the destination
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23)
	//则r0=max(m10,m20), r1=max(m11,m21), r2=max(m12,m22), r3=max(m13,m23)
	extern __m256d __cdecl _mm256_max_pd(__m256d m1, __m256d m2);

	/*
	* Maximum of Packed Single Precision Floating-Point Values
	* **** VMAXPS ymm1, ymm2, ymm3/m256
	* Performs an SIMD compare of the packed single-precision floating-point
	* values in the first source operand and the second source operand and
	* returns the maximum value for each pair of values to the destination
	*/
	//m1=(m10, m11, ..., m17), m2=(m20, m21, ..., m27)
	//则r0=max(m10,m20), r1=max(m11,m21), ..., r6=max(m16,m26), r7=max(m17,m27)
	extern __m256 __cdecl _mm256_max_ps(__m256 m1, __m256 m2);

	/*
	* Minimum of Packed Double Precision Floating-Point Values
	* **** VMINPD ymm1, ymm2, ymm3/m256
	* Performs an SIMD compare of the packed double-precision floating-point
	* values in the first source operand and the second source operand and
	* returns the minimum value for each pair of values to the destination
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23)
	//则r0=min(m10,m20), r1=min(m11,m21), r2=min(m12,m22), r3=min(m13,m23)
	extern __m256d __cdecl _mm256_min_pd(__m256d m1, __m256d m2);

	/*
	* Minimum of Packed Single Precision Floating-Point Values
	* **** VMINPS ymm1, ymm2, ymm3/m256
	* Performs an SIMD compare of the packed single-precision floating-point
	* values in the first source operand and the second source operand and
	* returns the minimum value for each pair of values to the destination
	*/
	//m1=(m10, m11, ..., m17), m2=(m20, m21, ..., m27)
	//则r0=min(m10,m20), r1=min(m11,m21), ..., r6=min(m16,m26), r7=min(m17,m27)
	extern __m256 __cdecl _mm256_min_ps(__m256 m1, __m256 m2);

	/*
	* Multiply Packed Double Precision Floating-Point Values
	* **** VMULPD ymm1, ymm2, ymm3/m256
	* Performs a SIMD multiply of the four packed double-precision floating-point
	* values from the first Source operand to the Second Source operand, and
	* stores the packed double-precision floating-point results in the
	* destination
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23)
	//则r0=m10*m20, r1=m11*m21, r2=m12*m22, r3=m13*m23
	extern __m256d __cdecl _mm256_mul_pd(__m256d m1, __m256d m2);

	/*
	* Multiply Packed Single Precision Floating-Point Values
	* **** VMULPS ymm1, ymm2, ymm3/m256
	* Performs an SIMD multiply of the eight packed single-precision
	* floating-point values from the first source operand to the second source
	* operand, and stores the packed double-precision floating-point results in
	* the destination
	*/
	//m1=(m10, m11, ..., m17), m2=(m20, m21, ..., m27)
	//则r0=m10*m20, r1=m11*m21, ..., r6=m16*m26, r7=m17*m27
	extern __m256 __cdecl _mm256_mul_ps(__m256 m1, __m256 m2);

	/*
	* Bitwise Logical OR of Packed Double Precision Floating-Point Values
	* **** VORPD ymm1, ymm2, ymm3/m256
	* Performs a bitwise logical OR of the four packed double-precision
	* floating-point values from the first source operand and the second
	* source operand, and stores the result in the destination
	*/
	//注意：有时得到的结果并不是m1和m2按位或的结果?
	extern __m256d __cdecl _mm256_or_pd(__m256d m1, __m256d m2);

	/*
	* Bitwise Logical OR of Packed Single Precision Floating-Point Values
	* **** VORPS ymm1, ymm2, ymm3/m256
	* Performs a bitwise logical OR of the eight packed single-precision
	* floating-point values from the first source operand and the second
	* source operand, and stores the result in the destination
	*/
	//注意：有时得到的结果并不是m1和m2按位或的结果?
	extern __m256 __cdecl _mm256_or_ps(__m256 m1, __m256 m2);

	/*
	* Shuffle Packed Double Precision Floating-Point Values
	* **** VSHUFPD ymm1, ymm2, ymm3/m256, imm8
	* Moves either of the two packed double-precision floating-point values from
	* each double quadword in the first source operand into the low quadword
	* of each double quadword of the destination; moves either of the two packed
	* double-precision floating-point values from the second source operand into
	* the high quadword of each double quadword of the destination operand.
	* The selector operand determines which values are moved to the destination
	*/
	extern __m256d __cdecl _mm256_shuffle_pd(__m256d m1, __m256d m2, const int select);

	/*
	* Shuffle Packed Single Precision Floating-Point Values
	* **** VSHUFPS ymm1, ymm2, ymm3/m256, imm8
	* Moves two of the four packed single-precision floating-point values
	* from each double qword of the first source operand into the low
	* quadword of each double qword of the destination; moves two of the four
	* packed single-precision floating-point values from each double qword of
	* the second source operand into to the high quadword of each double qword
	* of the destination. The selector operand determines which values are moved
	* to the destination.
	*/
	extern __m256 __cdecl _mm256_shuffle_ps(__m256 m1, __m256 m2, const int select);

	/*
	* Subtract Packed Double Precision Floating-Point Values
	* **** VSUBPD ymm1, ymm2, ymm3/m256
	* Performs an SIMD subtract of the four packed double-precision floating-point
	* values of the second Source operand from the first Source operand, and
	* stores the packed double-precision floating-point results in the destination
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23)
	//则r0=m10-m20, r1=m11-m21, r2=m12-m22, r3=m13-m23
	extern __m256d __cdecl _mm256_sub_pd(__m256d m1, __m256d m2);

	/*
	* Subtract Packed Single Precision Floating-Point Values
	* **** VSUBPS ymm1, ymm2, ymm3/m256
	* Performs an SIMD subtract of the eight packed single-precision
	* floating-point values in the second Source operand from the First Source
	* operand, and stores the packed single-precision floating-point results in
	* the destination
	*/
	//m1=(m10, m11, ..., m17), m2=(m20, m21, ..., m27)
	//则r0=m10-m20, r1=m11-m21, ..., r6=m16-m26, r7=m17-m27
	extern __m256 __cdecl _mm256_sub_ps(__m256 m1, __m256 m2);

	/*
	* Bitwise Logical XOR of Packed Double Precision Floating-Point Values
	* **** VXORPD ymm1, ymm2, ymm3/m256
	* Performs a bitwise logical XOR of the four packed double-precision
	* floating-point values from the first source operand and the second
	* source operand, and stores the result in the destination
	*/
	//注意：有时得到的结果并不是m1和m2按位异或的结果?
	extern __m256d __cdecl _mm256_xor_pd(__m256d m1, __m256d m2);

	/*
	* Bitwise Logical XOR of Packed Single Precision Floating-Point Values
	* **** VXORPS ymm1, ymm2, ymm3/m256
	* Performs a bitwise logical XOR of the eight packed single-precision
	* floating-point values from the first source operand and the second
	* source operand, and stores the result in the destination
	*/
	//注意：有时得到的结果并不是m1和m2按位异或的结果?
	extern __m256 __cdecl _mm256_xor_ps(__m256 m1, __m256 m2);

	/*
	* Compare Packed Double-Precision Floating-Point Values
	* **** VCMPPD xmm1, xmm2, xmm3/m128, imm8
	* **** VCMPPD ymm1, ymm2, ymm3/m256, imm8
	* Performs an SIMD compare of the four packed double-precision floating-point
	* values in the second source operand (third operand) and the first source
	* operand (second operand) and returns the results of the comparison to the
	* destination operand (first operand). The comparison predicate operand
	* (immediate) specifies the type of comparison performed on each of the pairs
	* of packed values.
	* For 128-bit intrinsic with compare predicate values in range 0-7 compiler
	* may generate SSE2 instructions if it is warranted for performance reasons.
	*/
	extern __m128d __cdecl _mm_cmp_pd(__m128d m1, __m128d m2, const int predicate);
	extern __m256d __cdecl _mm256_cmp_pd(__m256d m1, __m256d m2, const int predicate);

	/*
	* Compare Packed Single-Precision Floating-Point Values
	* **** VCMPPS xmm1, xmm2, xmm3/m256, imm8
	* **** VCMPPS ymm1, ymm2, ymm3/m256, imm8
	* Performs a SIMD compare of the packed single-precision floating-point values
	* in the second source operand (third operand) and the first source operand
	* (second operand) and returns the results of the comparison to the destination
	* operand (first operand). The comparison predicate operand (immediate)
	* specifies the type of comparison performed on each of the pairs of packed
	* values.
	* For 128-bit intrinsic with compare predicate values in range 0-7 compiler
	* may generate SSE2 instructions if it is warranted for performance reasons.
	*/
	extern __m128 __cdecl _mm_cmp_ps(__m128 m1, __m128 m2, const int predicate);
	extern __m256 __cdecl _mm256_cmp_ps(__m256 m1, __m256 m2, const int predicate);

	/*
	* Compare Scalar Double-Precision Floating-Point Values
	* **** VCMPSD xmm1, xmm2, xmm3/m64, imm8
	* Compares the low double-precision floating-point values in the second source
	* operand (third operand) and the first source operand (second operand) and
	* returns the results in of the comparison to the destination operand (first
	* operand). The comparison predicate operand (immediate operand) specifies the
	* type of comparison performed.
	* For compare predicate values in range 0-7 compiler may generate SSE2
	* instructions if it is warranted for performance reasons.
	*/
	extern __m128d __cdecl _mm_cmp_sd(__m128d m1, __m128d m2, const int predicate);

	/*
	* Compare Scalar Single-Precision Floating-Point Values
	* **** VCMPSS xmm1, xmm2, xmm3/m64, imm8
	* Compares the low single-precision floating-point values in the second source
	* operand (third operand) and the first source operand (second operand) and
	* returns the results of the comparison to the destination operand (first
	* operand). The comparison predicate operand (immediate operand) specifies
	* the type of comparison performed.
	* For compare predicate values in range 0-7 compiler may generate SSE2
	* instructions if it is warranted for performance reasons.
	*/
	extern __m128 __cdecl _mm_cmp_ss(__m128 m1, __m128 m2, const int predicate);

	/*
	* Convert Packed Doubleword Integers to
	* Packed Double-Precision Floating-Point Values
	* **** VCVTDQ2PD ymm1, xmm2/m128
	* Converts four packed signed doubleword integers in the source operand to
	* four packed double-precision floating-point values in the destination
	*/
	//从__int32类型转换到double类型
	extern __m256d __cdecl _mm256_cvtepi32_pd(__m128i m1);

	/*
	* Convert Packed Doubleword Integers to
	* Packed Single-Precision Floating-Point Values
	* **** VCVTDQ2PS ymm1, ymm2/m256
	* Converts eight packed signed doubleword integers in the source operand to
	* eight packed double-precision floating-point values in the destination
	*/
	//从__int32类型转换到float类型
	extern __m256  __cdecl _mm256_cvtepi32_ps(__m256i m1);

	/*
	* Convert Packed Double-Precision Floating-point values to
	* Packed Single-Precision Floating-Point Values
	* **** VCVTPD2PS xmm1, ymm2/m256
	* Converts four packed double-precision floating-point values in the source
	* operand to four packed single-precision floating-point values in the
	* destination
	*/
	//从double类型转换到float类型
	extern __m128  __cdecl _mm256_cvtpd_ps(__m256d m1);

	/*
	* Convert Packed Single Precision Floating-Point Values to
	* Packed Singed Doubleword Integer Values
	* **** VCVTPS2DQ ymm1, ymm2/m256
	* Converts eight packed single-precision floating-point values in the source
	* operand to eight signed doubleword integers in the destination
	*/
	//从float类型转换到__int32类型
	extern __m256i __cdecl _mm256_cvtps_epi32(__m256 m1);

	/*
	* Convert Packed Single Precision Floating-point values to
	* Packed Double Precision Floating-Point Values
	* **** VCVTPS2PD ymm1, xmm2/m128
	* Converts four packed single-precision floating-point values in the source
	* operand to four packed double-precision floating-point values in the
	* destination
	*/
	//从float类型转换到double类型
	extern __m256d __cdecl _mm256_cvtps_pd(__m128 m1);

	/*
	* Convert with Truncation Packed Double-Precision Floating-Point values to
	* Packed Doubleword Integers
	* **** VCVTTPD2DQ xmm1, ymm2/m256
	* Converts four packed double-precision floating-point values in the source
	* operand to four packed signed doubleword integers in the destination.
	* When a conversion is inexact, a truncated (round toward zero) value is
	* returned. If a converted result is larger than the maximum signed doubleword
	* integer, the floating-point invalid exception is raised, and if this
	* exception is masked, the indefinite integer value (80000000H) is returned
	*/
	//从double类型转换到__int32类型,truncated
	extern __m128i __cdecl _mm256_cvttpd_epi32(__m256d m1);

	/*
	* Convert Packed Double-Precision Floating-point values to
	* Packed Doubleword Integers
	* **** VCVTPD2DQ xmm1, ymm2/m256
	* Converts four packed double-precision floating-point values in the source
	* operand to four packed signed doubleword integers in the destination
	*/
	//从double类型转换到__int32类型
	extern __m128i __cdecl _mm256_cvtpd_epi32(__m256d m1);

	/*
	* Convert with Truncation Packed Single Precision Floating-Point Values to
	* Packed Singed Doubleword Integer Values
	* **** VCVTTPS2DQ ymm1, ymm2/m256
	* Converts eight packed single-precision floating-point values in the source
	* operand to eight signed doubleword integers in the destination.
	* When a conversion is inexact, a truncated (round toward zero) value is
	* returned. If a converted result is larger than the maximum signed doubleword
	* integer, the floating-point invalid exception is raised, and if this
	* exception is masked, the indefinite integer value (80000000H) is returned
	*/
	//从float类型转换到__int32类型,truncated
	extern __m256i __cdecl _mm256_cvttps_epi32(__m256 m1);

	/*
	* Extract packed floating-point values
	* **** VEXTRACTF128 xmm1/m128, ymm2, imm8
	* Extracts 128-bits of packed floating-point values from the source operand
	* at an 128-bit offset from imm8[0] into the destination
	*/
	//offset:a constant integer value that represents the 128-bit offset from
	//where extraction must start
	//从256位中提取128位,offset决定提取的起始位置
	extern __m128  __cdecl _mm256_extractf128_ps(__m256 m1, const int offset);
	extern __m128d __cdecl _mm256_extractf128_pd(__m256d m1, const int offset);
	extern __m128i __cdecl _mm256_extractf128_si256(__m256i m1, const int offset);

	/*
	* Zero All YMM registers
	* **** VZEROALL
	* Zeros contents of all YMM registers
	*/
	extern void __cdecl _mm256_zeroall(void);

	/*
	* Zero Upper bits of YMM registers
	* **** VZEROUPPER
	* Zeros the upper 128 bits of all YMM registers. The lower 128-bits of the
	* registers (the corresponding XMM registers) are unmodified
	*/
	extern void __cdecl _mm256_zeroupper(void);

	/*
	* Permute Single-Precision Floating-Point Values
	* **** VPERMILPS ymm1, ymm2, ymm3/m256
	* **** VPERMILPS xmm1, xmm2, xmm3/m128
	* Permute Single-Precision Floating-Point values in the first source operand
	* using 8-bit control fields in the low bytes of corresponding elements the
	* shuffle control and store results in the destination
	*/
	//control:a vector with 2-bit control fields, one for each corresponding element
	//of the source vector, for the 256-bit m1 source vector this control vector
	//contains eight 2-bit control fields,for the 128-bit m1 source vector this
	//control vector contains four 2-bit control fields
	extern __m256  __cdecl _mm256_permutevar_ps(__m256 m1, __m256i control);
	extern __m128  __cdecl _mm_permutevar_ps(__m128 a, __m128i control);

	/*
	* Permute Single-Precision Floating-Point Values
	* **** VPERMILPS ymm1, ymm2/m256, imm8
	* **** VPERMILPS xmm1, xmm2/m128, imm8
	* Permute Single-Precision Floating-Point values in the first source operand
	* using four 2-bit control fields in the 8-bit immediate and store results
	* in the destination
	*/
	//control:an integer specified as an 8-bit immediate;for the 256-bit m1 vector
	//this integer contains four 2-bit control fields in the low 8 bits of
	//the immediate, for the 128-bit m1 vector this integer contains two 2-bit
	//control fields in the low 4 bits of the immediate
	extern __m256  __cdecl _mm256_permute_ps(__m256 m1, int control);
	extern __m128  __cdecl _mm_permute_ps(__m128 a, int control);

	/*
	* Permute Double-Precision Floating-Point Values
	* **** VPERMILPD ymm1, ymm2, ymm3/m256
	* **** VPERMILPD xmm1, xmm2, xmm3/m128
	* Permute Double-Precision Floating-Point values in the first source operand
	* using 8-bit control fields in the low bytes of the second source operand
	* and store results in the destination
	*/
	//control:a vector with 1-bit control fields, one for each corresponding element
	//of the source vector, for the 256-bit m1 source vector this control vector
	//contains four 1-bit control fields in the low 4 bits of the immediate, for the
	//128-bit m1 source vector this control vector contains two 1-bit control fields
	//in the low 2 bits of the immediate
	extern __m256d __cdecl _mm256_permutevar_pd(__m256d m1, __m256i control);
	extern __m128d __cdecl _mm_permutevar_pd(__m128d a, __m128i control);

	/*
	* Permute Double-Precision Floating-Point Values
	* **** VPERMILPD ymm1, ymm2/m256, imm8
	* **** VPERMILPD xmm1, xmm2/m128, imm8
	* Permute Double-Precision Floating-Point values in the first source operand
	* using two, 1-bit control fields in the low 2 bits of the 8-bit immediate
	* and store results in the destination
	*/
	//control:an integer specified as an 8-bit immediate; for the 256-bit m1 vector
	//this integer contains four 1-bit control fields in the low 4 bits of the
	//immediate, for the 128-bit m1 vector this integer contains two 1-bit
	//control fields in the low 2 bits of the immediate
	extern __m256d __cdecl _mm256_permute_pd(__m256d m1, int control);
	extern __m128d __cdecl _mm_permute_pd(__m128d a, int control);

	/*
	* Permute Floating-Point Values
	* **** VPERM2F128 ymm1, ymm2, ymm3/m256, imm8
	* Permute 128 bit floating-point-containing fields from the first source
	* operand and second source operand using bits in the 8-bit immediate and
	* store results in the destination
	*/
	//control:an immediate byte that specifies two 2-bit control fields and two
	//additional bits which specify zeroing behavior
	extern __m256  __cdecl _mm256_permute2f128_ps(__m256 m1, __m256 m2, int control);
	extern __m256d __cdecl _mm256_permute2f128_pd(__m256d m1, __m256d m2, int control);
	extern __m256i __cdecl _mm256_permute2f128_si256(__m256i m1, __m256i m2, int control);

	/*
	* Load with Broadcast
	* **** VBROADCASTSS ymm1, m32
	* **** VBROADCASTSS xmm1, m32
	* Load floating point values from the source operand and broadcast to all
	* elements of the destination
	*/
	//*a:pointer to a memory location that can hold constant 256-bit or
	//128-bit float32 values, 则r0=r1=...=rn=a[0]
	extern __m256  __cdecl _mm256_broadcast_ss(float const *a);
	extern __m128  __cdecl _mm_broadcast_ss(float const *a);

	/*
	* Load with Broadcast
	* **** VBROADCASTSD ymm1, m64
	* Load floating point values from the source operand and broadcast to all
	* elements of the destination
	*/
	//则r0=r1=r2=r3=a[0]
	extern __m256d __cdecl _mm256_broadcast_sd(double const *a);

	/*
	* Load with Broadcast
	* **** VBROADCASTF128 ymm1, m128
	* Load floating point values from the source operand and broadcast to all
	* elements of the destination
	*/
	//若*a为a[0],a[1],则r0=r2=a[0], r1=r3=a[1]
	extern __m256  __cdecl _mm256_broadcast_ps(__m128 const *a);
	extern __m256d __cdecl _mm256_broadcast_pd(__m128d const *a);

	/*
	* Insert packed floating-point values
	* **** VINSERTF128 ymm1, ymm2, xmm3/m128, imm8
	* Performs an insertion of 128-bits of packed floating-point values from the
	* second source operand into an the destination at an 128-bit offset from
	* imm8[0]. The remaining portions of the destination are written by the
	* corresponding fields of the first source operand
	*/
	//offset:an integer value that represents the 128-bit offset
	//where the insertion must start
	//The remaining portions of the destination are written by the corresponding
	//elements of the first source vector, a
	extern __m256  __cdecl _mm256_insertf128_ps(__m256 a, __m128 b, int offset);
	extern __m256d __cdecl _mm256_insertf128_pd(__m256d a, __m128d b, int offset);
	extern __m256i __cdecl _mm256_insertf128_si256(__m256i a, __m128i b, int offset);

	/*
	* Move Aligned Packed Double-Precision Floating-Point Values
	* **** VMOVAPD ymm1, m256
	* **** VMOVAPD m256, ymm1
	* Moves 4 double-precision floating-point values from the source operand to
	* the destination
	*/
	//*a:the address must be 32-byte aligned
	extern __m256d __cdecl _mm256_load_pd(double const *a);
	extern void    __cdecl _mm256_store_pd(double *a, __m256d b);

	/*
	* Move Aligned Packed Single-Precision Floating-Point Values
	* **** VMOVAPS ymm1, m256
	* **** VMOVAPS m256, ymm1
	* Moves 8 single-precision floating-point values from the source operand to
	* the destination
	*/
	//*a:the address must be 32-byte aligned
	extern __m256  __cdecl _mm256_load_ps(float const *a);
	extern void    __cdecl _mm256_store_ps(float *a, __m256 b);

	/*
	* Move Unaligned Packed Double-Precision Floating-Point Values
	* **** VMOVUPD ymm1, m256
	* **** VMOVUPD m256, ymm1
	* Moves 256 bits of packed double-precision floating-point values from the
	* source operand to the destination
	*/
	//The address a does not need to be 32-byte aligned
	extern __m256d __cdecl _mm256_loadu_pd(double const *a);
	extern void    __cdecl _mm256_storeu_pd(double *a, __m256d b);

	/*
	* Move Unaligned Packed Single-Precision Floating-Point Values
	* **** VMOVUPS ymm1, m256
	* **** VMOVUPS m256, ymm1
	* Moves 256 bits of packed single-precision floating-point values from the
	* source operand to the destination
	*/
	//The address a does not need to be 32-byte aligned
	extern __m256  __cdecl _mm256_loadu_ps(float const *a);
	extern void    __cdecl _mm256_storeu_ps(float *a, __m256 b);

	/*
	* Move Aligned Packed Integer Values
	* **** VMOVDQA ymm1, m256
	* **** VMOVDQA m256, ymm1
	* Moves 256 bits of packed integer values from the source operand to the
	* destination
	*/
	//The address a does not need to be 32-byte aligned
	extern __m256i __cdecl _mm256_load_si256(__m256i const *a);
	extern void    __cdecl _mm256_store_si256(__m256i *a, __m256i b);

	/*
	* Move Unaligned Packed Integer Values
	* **** VMOVDQU ymm1, m256
	* **** VMOVDQU m256, ymm1
	* Moves 256 bits of packed integer values from the source operand to the
	* destination
	*/
	//The address a does not need to be 32-byte aligned
	extern __m256i __cdecl _mm256_loadu_si256(__m256i const *a);
	extern void    __cdecl _mm256_storeu_si256(__m256i *a, __m256i b);

	/*
	* Conditional SIMD Packed Loads and Stores
	* **** VMASKMOVPD xmm1, xmm2, m128
	* **** VMASKMOVPD ymm1, ymm2, m256
	* **** VMASKMOVPD m128, xmm1, xmm2
	* **** VMASKMOVPD m256, ymm1, ymm2
	*
	* Load forms:
	* Load packed values from the 128-bit (XMM forms) or 256-bit (YMM forms)
	* memory location (third operand) into the destination XMM or YMM register
	* (first operand) using a mask in the first source operand (second operand).
	*
	* Store forms:
	* Stores packed values from the XMM or YMM register in the second source
	* operand (third operand) into the 128-bit (XMM forms) or 256-bit (YMM forms)
	* memory location using a mask in first source operand (second operand).
	* Stores are atomic.
	*/
	//The mask is calculated from the most significant bit of each qword of the mask
	//register. If any of the bits of the mask is set to zero, the corresponding value
	//from the memory location is not loaded, and the corresponding field of the
	//destination vector is set to zero.
	extern __m256d __cdecl _mm256_maskload_pd(double const *a, __m256i mask);
	extern void    __cdecl _mm256_maskstore_pd(double *a, __m256i mask, __m256d b);
	extern __m128d __cdecl _mm_maskload_pd(double const *a, __m128i mask);
	extern void    __cdecl _mm_maskstore_pd(double *a, __m128i mask, __m128d b);

	/*
	* Conditional SIMD Packed Loads and Stores
	* **** VMASKMOVPS xmm1, xmm2, m128
	* **** VMASKMOVPS ymm1, ymm2, m256
	* **** VMASKMOVPS m128, xmm1, xmm2
	* **** VMASKMOVPS m256, ymm1, ymm2
	*
	* Load forms:
	* Load packed values from the 128-bit (XMM forms) or 256-bit (YMM forms)
	* memory location (third operand) into the destination XMM or YMM register
	* (first operand) using a mask in the first source operand (second operand).
	*
	* Store forms:
	* Stores packed values from the XMM or YMM register in the second source
	* operand (third operand) into the 128-bit (XMM forms) or 256-bit (YMM forms)
	* memory location using a mask in first source operand (second operand).
	* Stores are atomic.
	*/
	//The mask is calculated from the most significant bit of each dword of the mask
	//register. If any of the bits of the mask is set to zero, the corresponding
	//value from the memory location is not loaded, and the corresponding field of
	//the destination vector is set to zero.
	extern __m256  __cdecl _mm256_maskload_ps(float const *a, __m256i mask);
	extern void    __cdecl _mm256_maskstore_ps(float *a, __m256i mask, __m256 b);
	extern __m128  __cdecl _mm_maskload_ps(float const *a, __m128i mask);
	extern void    __cdecl _mm_maskstore_ps(float *a, __m128i mask, __m128 b);

	/*
	* Replicate Single-Precision Floating-Point Values
	* **** VMOVSHDUP ymm1, ymm2/m256
	* Duplicates odd-indexed single-precision floating-point values from the
	* source operand
	*/
	//a=(a0, a1, a2, a3, a4, a5, a6, a7);则r=(a1, a1, a3, a3, a5, a5, a7, a7)
	extern __m256  __cdecl _mm256_movehdup_ps(__m256 a);

	/*
	* Replicate Single-Precision Floating-Point Values
	* **** VMOVSLDUP ymm1, ymm2/m256
	* Duplicates even-indexed single-precision floating-point values from the
	* source operand
	*/
	//a=(a0, a1, a2, a3, a4, a5, a6, a7);则r=(a0, a0, a2, a2, a4, a4, a6, a6)
	extern __m256  __cdecl _mm256_moveldup_ps(__m256 a);

	/*
	* Replicate Double-Precision Floating-Point Values
	* **** VMOVDDUP ymm1, ymm2/m256
	* Duplicates even-indexed double-precision floating-point values from the
	* source operand
	*/
	//a=(a0, a1, a2, a3), 则r=(a0, a0, a2, a2)
	extern __m256d __cdecl _mm256_movedup_pd(__m256d a);

	/*
	* Move Unaligned Integer
	* **** VLDDQU ymm1, m256
	* The instruction is functionally similar to VMOVDQU YMM, m256 for loading
	* from memory. That is: 32 bytes of data starting at an address specified by
	* the source memory operand are fetched from memory and placed in a
	* destination
	*/
	//*a:points to a memory location from where unaligned integer value must be moved
	extern __m256i __cdecl _mm256_lddqu_si256(__m256i const *a);

	/*
	* Store Packed Integers Using Non-Temporal Hint
	* **** VMOVNTDQ m256, ymm1
	* Moves the packed integers in the source operand to the destination using a
	* non-temporal hint to prevent caching of the data during the write to memory
	*/
	//the address must be 32-byte aligned
	extern void    __cdecl _mm256_stream_si256(__m256i *p, __m256i a);

	/*
	* Store Packed Double-Precision Floating-Point Values Using Non-Temporal Hint
	* **** VMOVNTPD m256, ymm1
	* Moves the packed double-precision floating-point values in the source
	* operand to the destination operand using a non-temporal hint to prevent
	* caching of the data during the write to memory
	*/
	//the address must be 32-byte aligned
	extern void    __cdecl _mm256_stream_pd(double *p, __m256d a);

	/*
	* Store Packed Single-Precision Floating-Point Values Using Non-Temporal Hint
	* **** VMOVNTPS m256, ymm1
	* Moves the packed single-precision floating-point values in the source
	* operand to the destination operand using a non-temporal hint to prevent
	* caching of the data during the write to memory
	*/
	//the address must be 32-byte aligned
	extern void    __cdecl _mm256_stream_ps(float *p, __m256 a);

	/*
	* Compute Approximate Reciprocals of Packed Single-Precision Floating-Point Values
	* **** VRCPPS ymm1, ymm2/m256
	* Performs an SIMD computation of the approximate reciprocals of the eight
	* packed single precision floating-point values in the source operand and
	* stores the packed single-precision floating-point results in the destination
	*/
	//a=(a0, a1, a2, ..., a6, a7);
	//则r=(1/a0, 1/a1, ..., 1/a6, 1/a7), 求倒数
	extern __m256  __cdecl _mm256_rcp_ps(__m256 a);

	/*
	* Compute Approximate Reciprocals of Square Roots of
	* Packed Single-Precision Floating-point Values
	* **** VRSQRTPS ymm1, ymm2/m256
	* Performs an SIMD computation of the approximate reciprocals of the square
	* roots of the eight packed single precision floating-point values in the
	* source operand and stores the packed single-precision floating-point results
	* in the destination
	*/
	//a=(a0, a1, a2, ..., a6, a7);
	//则r=(1/sqrt(a0), 1/sqrt(a1), ..., 1/sqrt(a6), 1/sqrt(a7)), 先开方再求倒数
	extern __m256  __cdecl _mm256_rsqrt_ps(__m256 a);

	/*
	* Square Root of Double-Precision Floating-Point Values
	* **** VSQRTPD ymm1, ymm2/m256
	* Performs an SIMD computation of the square roots of the two or four packed
	* double-precision floating-point values in the source operand and stores
	* the packed double-precision floating-point results in the destination
	*/
	//a=(a0, a1, a2, a3, a4);则r=(sqrt(a0),sqrt(a1), sqrt(a2), sqrt(a3)), 求开方
	extern __m256d __cdecl _mm256_sqrt_pd(__m256d a);

	/*
	* Square Root of Single-Precision Floating-Point Values
	* **** VSQRTPS ymm1, ymm2/m256
	* Performs an SIMD computation of the square roots of the eight packed
	* single-precision floating-point values in the source operand stores the
	* packed double-precision floating-point results in the destination
	*/
	//a=(a0, a1, a2, ..., a3, a4);则r=(sqrt(a0),sqrt(a1), ..., sqrt(a2), sqrt(a3)), 求开方
	extern __m256  __cdecl _mm256_sqrt_ps(__m256 a);

	/*
	* Round Packed Double-Precision Floating-Point Values
	* **** VROUNDPD ymm1,ymm2/m256,imm8
	* Round the four Double-Precision Floating-Point Values values in the source
	* operand by the rounding mode specified in the immediate operand and place
	* the result in the destination. The rounding process rounds the input to an
	* integral value and returns the result as a double-precision floating-point
	* value. The Precision Floating Point Exception is signaled according to the
	* immediate operand. If any source operand is an SNaN then it will be
	* converted to a QNaN.
	*/
	//a=(22.8, -11.3, -33.8, 4.3),
	//若iRoundMode=0x0A, 则r=(23, -11, -33, 5)
	//若iRoundMode=0x09, 则r=(22, -12, -34, 4)
	extern __m256d __cdecl _mm256_round_pd(__m256d a, int iRoundMode);
#define _mm256_ceil_pd(val)   _mm256_round_pd((val), 0x0A);
#define _mm256_floor_pd(val)  _mm256_round_pd((val), 0x09);

	/*
	* Round Packed Single-Precision Floating-Point Values
	* **** VROUNDPS ymm1,ymm2/m256,imm8
	* Round the four single-precision floating-point values values in the source
	* operand by the rounding mode specified in the immediate operand and place
	* the result in the destination. The rounding process rounds the input to an
	* integral value and returns the result as a double-precision floating-point
	* value. The Precision Floating Point Exception is signaled according to the
	* immediate operand. If any source operand is an SNaN then it will be
	* converted to a QNaN.
	*/
	//用法与_mm256_round_pd相同
	extern __m256  __cdecl _mm256_round_ps(__m256 a, int iRoundMode);
#define _mm256_ceil_ps(val)   _mm256_round_ps((val), 0x0A);
#define _mm256_floor_ps(val)  _mm256_round_ps((val), 0x09);

	/*
	* Unpack and Interleave High Packed Double-Precision Floating-Point Values
	* **** VUNPCKHPD ymm1,ymm2,ymm3/m256
	* Performs an interleaved unpack of the high double-precision floating-point
	* values from the first source operand and the second source operand.
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23)
	//则r=(m11, m21, m13, m23)
	extern __m256d __cdecl _mm256_unpackhi_pd(__m256d m1, __m256d m2);

	/*
	* Unpack and Interleave High Packed Single-Precision Floating-Point Values
	* **** VUNPCKHPS ymm1,ymm2,ymm3
	* Performs an interleaved unpack of the high single-precision floating-point
	* values from the first source operand and the second source operand
	*/
	//m1=(m10, m11, ..., m17), m2=(m20, m21, ..., m27)
	//则r=(m12, m22, m13, m23, m16, m26, m17, m27)
	extern __m256  __cdecl _mm256_unpackhi_ps(__m256 m1, __m256 m2);

	/*
	* Unpack and Interleave Low Packed Double-Precision Floating-Point Values
	* **** VUNPCKLPD ymm1,ymm2,ymm3/m256
	* Performs an interleaved unpack of the low double-precision floating-point
	* values from the first source operand and the second source operand
	*/
	//m1=(m10, m11, m12, m13), m2=(m20, m21, m22, m23)
	//则r=(m10, m20, m12, m22)
	extern __m256d __cdecl _mm256_unpacklo_pd(__m256d m1, __m256d m2);

	/*
	* Unpack and Interleave Low Packed Single-Precision Floating-Point Values
	* **** VUNPCKLPS ymm1,ymm2,ymm3
	* Performs an interleaved unpack of the low single-precision floating-point
	* values from the first source operand and the second source operand
	*/
	//m1=(m10, m11, ..., m17), m2=(m20, m21, ..., m27)
	//则r=(m10, m20, m11, m21, m14, m24, m15, m25)
	extern __m256  __cdecl _mm256_unpacklo_ps(__m256 m1, __m256 m2);

	/*
	* Packed Bit Test
	* **** VPTEST ymm1, ymm2/m256
	* VPTEST set the ZF flag if all bits in the result are 0 of the bitwise AND
	* of the first source operand and the second source operand. VPTEST sets the
	* CF flag if all bits in the result are 0 of the bitwise AND of the second
	* source operand and the logical NOT of the destination.
	*/
	extern int     __cdecl _mm256_testz_si256(__m256i s1, __m256i s2);
	extern int     __cdecl _mm256_testc_si256(__m256i s1, __m256i s2);
	extern int     __cdecl _mm256_testnzc_si256(__m256i s1, __m256i s2);

	/*
	* Packed Bit Test
	* **** VTESTPD ymm1, ymm2/m256
	* **** VTESTPD xmm1, xmm2/m128
	* VTESTPD performs a bitwise comparison of all the sign bits of the
	* double-precision elements in the first source operation and corresponding
	* sign bits in the second source operand. If the AND of the two sets of bits
	* produces all zeros, the ZF is set else the ZF is clear. If the AND NOT of
	* the source sign bits with the dest sign bits produces all zeros the CF is
	* set else the CF is clear
	*/
	extern int     __cdecl _mm256_testz_pd(__m256d s1, __m256d s2);
	extern int     __cdecl _mm256_testc_pd(__m256d s1, __m256d s2);
	extern int     __cdecl _mm256_testnzc_pd(__m256d s1, __m256d s2);
	extern int     __cdecl _mm_testz_pd(__m128d s1, __m128d s2);
	extern int     __cdecl _mm_testc_pd(__m128d s1, __m128d s2);
	extern int     __cdecl _mm_testnzc_pd(__m128d s1, __m128d s2);

	/*
	* Packed Bit Test
	* **** VTESTPS ymm1, ymm2/m256
	* **** VTESTPS xmm1, xmm2/m128
	* VTESTPS performs a bitwise comparison of all the sign bits of the packed
	* single-precision elements in the first source operation and corresponding
	* sign bits in the second source operand. If the AND of the two sets of bits
	* produces all zeros, the ZF is set else the ZF is clear. If the AND NOT of
	* the source sign bits with the dest sign bits produces all zeros the CF is
	* set else the CF is clear
	*/
	extern int     __cdecl _mm256_testz_ps(__m256 s1, __m256 s2);
	extern int     __cdecl _mm256_testc_ps(__m256 s1, __m256 s2);
	extern int     __cdecl _mm256_testnzc_ps(__m256 s1, __m256 s2);
	extern int     __cdecl _mm_testz_ps(__m128 s1, __m128 s2);
	extern int     __cdecl _mm_testc_ps(__m128 s1, __m128 s2);
	extern int     __cdecl _mm_testnzc_ps(__m128 s1, __m128 s2);

	/*
	* Extract Double-Precision Floating-Point Sign mask
	* **** VMOVMSKPD r32, ymm2
	* Extracts the sign bits from the packed double-precision floating-point
	* values in the source operand, formats them into a 4-bit mask, and stores
	* the mask in the destination
	*/
	extern int     __cdecl _mm256_movemask_pd(__m256d a);

	/*
	* Extract Single-Precision Floating-Point Sign mask
	* **** VMOVMSKPS r32, ymm2
	* Extracts the sign bits from the packed single-precision floating-point
	* values in the source operand, formats them into a 8-bit mask, and stores
	* the mask in the destination
	*/
	extern int     __cdecl _mm256_movemask_ps(__m256 a);

	/*
	* Return 256-bit vector with all elements set to 0
	*/
	//则r0=r1=...=rn=0
	extern __m256d __cdecl _mm256_setzero_pd(void);
	extern __m256  __cdecl _mm256_setzero_ps(void);
	extern __m256i __cdecl _mm256_setzero_si256(void);

	/*
	* Return 256-bit vector intialized to specified arguments
	*/
	//则r = (d, c, b, a)
	extern __m256d __cdecl _mm256_set_pd(double a, double b, double c, double d);
	extern __m256  __cdecl _mm256_set_ps(float, float, float, float, float, float, float, float);
	extern __m256i __cdecl _mm256_set_epi8(char, char, char, char, char, char, char, char,
		char, char, char, char, char, char, char, char,
		char, char, char, char, char, char, char, char,
		char, char, char, char, char, char, char, char);
	extern __m256i __cdecl _mm256_set_epi16(short, short, short, short, short, short, short, short,
		short, short, short, short, short, short, short, short);
	extern __m256i __cdecl _mm256_set_epi32(int, int, int, int, int, int, int, int);
	extern __m256i __cdecl _mm256_set_epi64x(long long, long long, long long, long long);

	//则r = (a, b, c, d)
	extern __m256d __cdecl _mm256_setr_pd(double a, double b, double c, double d);
	extern __m256  __cdecl _mm256_setr_ps(float, float, float, float, float, float, float, float);
	extern __m256i __cdecl _mm256_setr_epi8(char, char, char, char, char, char, char, char,
		char, char, char, char, char, char, char, char,
		char, char, char, char, char, char, char, char,
		char, char, char, char, char, char, char, char);
	extern __m256i __cdecl _mm256_setr_epi16(short, short, short, short, short, short, short, short,
		short, short, short, short, short, short, short, short);
	extern __m256i __cdecl _mm256_setr_epi32(int, int, int, int, int, int, int, int);
	extern __m256i __cdecl _mm256_setr_epi64x(long long, long long, long long, long long);

	/*
	* Return 256-bit vector with all elements intialized to specified scalar
	*/
	//则r0 =  ... = rn = a
	extern __m256d __cdecl _mm256_set1_pd(double a);
	extern __m256  __cdecl _mm256_set1_ps(float);
	extern __m256i __cdecl _mm256_set1_epi8(char);
	extern __m256i __cdecl _mm256_set1_epi16(short);
	extern __m256i __cdecl _mm256_set1_epi32(int);
	extern __m256i __cdecl _mm256_set1_epi64x(long long);

	/*
	* Support intrinsics to do vector type casts. These intrinsics do not introduce
	* extra moves to generated code. When cast is done from a 128 to 256-bit type
	* the low 128 bits of the 256-bit result contain source parameter value; the
	* upper 128 bits of the result are undefined
	*/
	//类型转换
	extern __m256  __cdecl _mm256_castpd_ps(__m256d a);
	extern __m256d __cdecl _mm256_castps_pd(__m256 a);
	extern __m256i __cdecl _mm256_castps_si256(__m256 a);
	extern __m256i __cdecl _mm256_castpd_si256(__m256d a);
	extern __m256  __cdecl _mm256_castsi256_ps(__m256i a);
	extern __m256d __cdecl _mm256_castsi256_pd(__m256i a);
	extern __m128  __cdecl _mm256_castps256_ps128(__m256 a);
	extern __m128d __cdecl _mm256_castpd256_pd128(__m256d a);
	extern __m128i __cdecl _mm256_castsi256_si128(__m256i a);
	extern __m256  __cdecl _mm256_castps128_ps256(__m128 a);
	extern __m256d __cdecl _mm256_castpd128_pd256(__m128d a);
	extern __m256i __cdecl _mm256_castsi128_si256(__m128i a);

	/* Start of new intrinsics for Dev10 SP1
	*
	* The list of extended control registers.
	* Currently, the list includes only one register.
	*/
#define _XCR_XFEATURE_ENABLED_MASK 0

	/* Returns the content of the specified extended control register */
	extern unsigned __int64 __cdecl _xgetbv(unsigned int ext_ctrl_reg);

	/* Writes the value to the specified extended control register */
	extern void __cdecl _xsetbv(unsigned int ext_ctrl_reg, unsigned __int64 val);


	/* 
	* Performs a full or partial save of the enabled processor state components
	* using the the specified memory address location and a mask.
	*/
	extern void __cdecl _xsave(void *mem, unsigned __int64 save_mask);
	extern void __cdecl _xsave64(void *mem, unsigned __int64 save_mask);

	/* 
	* Performs a full or partial save of the enabled processor state components
	* using the the specified memory address location and a mask.
	* Optimize the state save operation if possible.
	*/
	extern void __cdecl _xsaveopt(void *mem, unsigned __int64 save_mask);
	extern void __cdecl _xsaveopt64(void *mem, unsigned __int64 save_mask);

	/* 
	* Performs a full or partial restore of the enabled processor states
	* using the state information stored in the specified memory address location
	* and a mask.
	*/
	extern void __cdecl _xrstor(void *mem, unsigned __int64 restore_mask);
	extern void __cdecl _xrstor64(void *mem, unsigned __int64 restore_mask);

	/* 
	* Saves the current state of the x87 FPU, MMX technology, XMM,
	* and MXCSR registers to the specified 512-byte memory location.
	*/
	extern void __cdecl _fxsave(void *mem);
	extern void __cdecl _fxsave64(void *mem);

	/* 
	* Restore the current state of the x87 FPU, MMX technology, XMM,
	* and MXCSR registers from the specified 512-byte memory location.
	*/
	extern void __cdecl _fxrstor(void *mem);
	extern void __cdecl _fxrstor64(void *mem);
}
