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

void MMXFUN()
{
	/* General support intrinsics */
	//Empties the multimedia state,清除MMX寄存器中的内容,即初始化(以避免和浮点数
	//操作发生冲突),详细说明见参考文献1
	void  _m_empty(void);//_mm_empty
	//Converts the integer object _I to a 64-bit __m64 object, r0=_I, r1=0
	__m64 _m_from_int(int _I);//_mm_cvtsi32_si64
	//Converts the lower 32 bits of the __m64 object _M to an integer, r=_M0
	int   _m_to_int(__m64 _M);//_mm_cvtsi64_si32
	//Packs the four 16-bit values from _MM1 into the lower four 8-bit values of
	//the result with signed saturation, and packs the four 16-bit values from _MM2
	//into the upper four 8-bit values of the result with signed saturation
	__m64 _m_packsswb(__m64 _MM1, __m64 _MM2);//_mm_packs_pi16
	//Packs the two 32-bit values from _MM1 into the lower two 16-bit values of the
	// result with signed saturation, and packs the two 32-bit values from _MM2 into
	// the upper two 16-bit values of the result with signed saturation
	__m64 _m_packssdw(__m64 _MM1, __m64 _MM2);//_mm_packs_pi32
	//Packs the four 16-bit values from _MM1 into the lower four 8-bit values of the
	//result with unsigned saturation, and packs the four 16-bit values from _MM2 into
	//the upper four 8-bit values of the result with unsigned saturation
	__m64 _m_packuswb(__m64 _MM1, __m64 _MM2);//_mm_packs_pu16
	//_MM1=(_MM10, _MM11, _MM12, _MM13, _MM14, _MM15, _MM16, _MM17),
	//_MM2=(_MM20, _MM21, _MM22, _MM23, _MM24, _MM25, _MM26, _MM27),
	//则r=(_MM14, _MM24, _MM15, _MM25, _MM16, _MM26, _MM17, _MM27)
	__m64 _m_punpckhbw(__m64 _MM1, __m64 _MM2);//_mm_unpackhi_pi8 
	//_MM1=(_MM10, _MM11, _MM12, _MM13),_MM10为低位,_MM2=(_MM20, _MM21, _MM22, _MM23),
	//则r=(_MM12, _MM22, _MM13, _MM23)
	__m64 _m_punpckhwd(__m64 _MM1, __m64 _MM2);//_mm_unpackhi_pi16
	//MM1=(_MM10, _MM11),_MM10为低位,_MM2=(_MM20, _MM21),则r=(_MM11, _MM21)
	__m64 _m_punpckhdq(__m64 _MM1, __m64 _MM2);//_mm_unpackhi_pi32
	//_MM1=(_MM10, _MM11, _MM12, _MM13, _MM14, _MM15, _MM16, _MM17),
	//_MM2=(_MM20, _MM21, _MM22, _MM23, _MM24, _MM25, _MM26, _MM27),
	//则r=(_MM10, _MM20, _MM11, _MM21, _MM12, _MM22, _MM13, _MM23)
	__m64 _m_punpcklbw(__m64 _MM1, __m64 _MM2);//_mm_unpacklo_pi8
	//_MM1=(_MM10, _MM11, _MM12, _MM13),_MM10为低位,_MM2=(_MM20, _MM21, _MM22, _MM23),
	//则r=(_MM10, _MM20, _MM11, _MM21)
	__m64 _m_punpcklwd(__m64 _MM1, __m64 _MM2);//_mm_unpacklo_pi16
	//MM1=(_MM10, _MM11),_MM10为低位,_MM2=(_MM20, _MM21),则r=(_MM10, _MM20)
	__m64 _m_punpckldq(__m64 _MM1, __m64 _MM2);//mm_unpacklo_pi32

	/* Packed arithmetic intrinsics */
	//Adds the eight 8-bit values in _MM1 to the eight 8-bit values in _MM2
	__m64 _m_paddb(__m64 _MM1, __m64 _MM2);//_mm_add_pi8
	//Adds the four 16-bit values in _MM1 to the four 16-bit values in _MM2
	__m64 _m_paddw(__m64 _MM1, __m64 _MM2);//_mm_add_pi16
	//Adds the two 32-bit values in _MM1 to the two 32-bit values in _MM2
	__m64 _m_paddd(__m64 _MM1, __m64 _MM2);//_mm_add_pi32
	//Adds the eight signed 8-bit values in _MM1 to the eight signed 8-bit values in _MM2
	//and saturates
	__m64 _m_paddsb(__m64 _MM1, __m64 _MM2);//_mm_adds_pi8
	//Adds the four signed 16-bit values in _MM1 to the four signed 16-bit values in _MM2
	//and saturates
	__m64 _m_paddsw(__m64 _MM1, __m64 _MM2);//_mm_adds_pi16
	//Adds the eight unsigned 8-bit values in _MM1 to the eight unsigned 8-bit values
	//in _MM2 and saturates
	__m64 _m_paddusb(__m64 _MM1, __m64 _MM2);//_mm_adds_pu8
	//Add the four unsigned 16-bit values in _MM1 to the four unsigned 16-bit values
	//in _MM2 and saturates
	__m64 _m_paddusw(__m64 _MM1, __m64 _MM2);//_mm_adds_pu16
	//Subtracts the eight 8-bit values in _MM2 from the eight 8-bit values in _MM1
	__m64 _m_psubb(__m64 _MM1, __m64 _MM2);//_mm_sub_pi8 
	//Subtracts the four 16-bit values in _MM2 from the four 16-bit values in _MM1
	__m64 _m_psubw(__m64 _MM1, __m64 _MM2);//_mm_sub_pi16
	//Subtracts the two 32-bit values in _MM2 from the two 32-bit values in _MM1
	__m64 _m_psubd(__m64 _MM1, __m64 _MM2);//_mm_sub_pi32
	//Subtracts the eight signed 8-bit values in _MM2 from the eight signed 8-bit
	//values in _MM1 and saturates
	__m64 _m_psubsb(__m64 _MM1, __m64 _MM2);//_mm_subs_pi8
	//Subtracts the four signed 16-bit values in _MM2 from the four signed 16-bit
	//values in _MM1 and saturates
	__m64 _m_psubsw(__m64 _MM1, __m64 _MM2);//_mm_subs_pi16
	//Subtracts the eight unsigned 8-bit values in _MM2 from the eight unsigned 8-bit
	//values in _MM1 and saturates
	__m64 _m_psubusb(__m64 _MM1, __m64 _MM2);//_mm_subs_pu8
	//Subtracts the four unsigned 16-bit values in _MM2 from the four unsigned 16-bit
	//values in _MM1 and saturates
	__m64 _m_psubusw(__m64 _MM1, __m64 _MM2);//_mm_subs_pu16
	//Multiplies four 16-bit values in _MM1 by four 16-bit values in _MM2 to produce
	//four 32-bit intermediate results, which are then summed by pairs to produce two
	//32-bit results,r0=_MM10*_MM20+_MM11*_MM21, r1=_MM12*_MM22+_MM13*_MM23
	__m64 _m_pmaddwd(__m64 _MM1, __m64 _MM2);//_mm_madd_pi16
	//Multiplies four signed 16-bit values in _MM1 by four signed 16-bit values in _MM2
	//and produces the high 16 bits of the four results
	__m64 _m_pmulhw(__m64 _MM1, __m64 _MM2);//_mm_mulhi_pi16
	//Multiplies four 16-bit values in _MM1 by four 16-bit values in _MM2 and produces
	//the low 16 bits of the four results
	__m64 _m_pmullw(__m64 _MM1, __m64 _MM2);//_mm_mullo_pi16

	/* Shift intrinsics */
	//Shifts four 16-bit values in _M left the amount specified by _Count
	//while shifting in zeros,左移_Count位,移出位补0
	__m64 _m_psllw(__m64 _M, __m64 _Count);//_mm_sll_pi16
	//Shifts four 16-bit values in _M left the amount specified by _Ccount while
	//shifting in zeros,左移_Count位,移出位补0,_Count需是一个立即数
	//汇编语言中的立即数相当于高级语言中的常量(常数),它是直接出现在指令中的数,
	//不用存储在寄存器或存储器中的数
	__m64 _m_psllwi(__m64 _M, int _Count);//_mm_slli_pi16 
	//Shifts two 32-bit values in _M left the amount specified by _Count
	//while shifting in zeros
	__m64 _m_pslld(__m64 _M, __m64 _Count);//_mm_sll_pi32
	//Shifts two 32-bit values in _M left the amount specified by _Count
	//while shifting in zeros
	__m64 _m_pslldi(__m64 _M, int _Count);//_mm_slli_pi32
	//Shifts the 64-bit value in _M left the amount specified by _Count
	//while shifting in zeros
	__m64 _m_psllq(__m64 _M, __m64 _Count);//_mm_sll_si64
	//Shifts the 64-bit value in _M left the amount specified by _Count
	//while shifting in zeros
	__m64 _m_psllqi(__m64 _M, int _Count);//_mm_slli_si64
	//Shifts four 16-bit values in _M right the amount specified by _Count
	//while shifting in the sign bit
	__m64 _m_psraw(__m64 _M, __m64 _Count);//_mm_sra_pi16
	//Shifts four 16-bit values in _M right the amount specified by _Count
	//while shifting in the sign bit
	__m64 _m_psrawi(__m64 _M, int _Count);//_mm_srai_pi16
	//Shifts two 32-bit values in _M right the amount specified by _Count
	//while shifting in the sign bit
	__m64 _m_psrad(__m64 _M, __m64 _Count);//_mm_sra_pi32
	//Shifts two 32-bit values in _M right the amount specified by _Count
	//while shifting in the sign bit
	__m64 _m_psradi(__m64 _M, int _Count);//_mm_srai_pi32
	//Shifts four 16-bit values in _M right the amount specified by _Count
	//while shifting in zeros
	__m64 _m_psrlw(__m64 _M, __m64 _Count);//_mm_srl_pi16
	//Shifts four 16-bit values in _M right the amount specified by _Count
	//while shifting in zeros
	__m64 _m_psrlwi(__m64 _M, int _Count);//_mm_srli_pi16
	//Shifts two 32-bit values in _M right the amount specified by _Count
	//while shifting in zeros
	__m64 _m_psrld(__m64 _M, __m64 _Count);//_mm_srl_pi32
	//Shifts two 32-bit values in _M right the amount specified by _Count
	//while shifting in zeros
	__m64 _m_psrldi(__m64 _M, int _Count);//_mm_srli_pi32 
	//Shifts the 64-bit value in _M right the amount specified by _Count
	//while shifting in zeros
	__m64 _m_psrlq(__m64 _M, __m64 _Count);//_mm_srl_si64
	//Shifts the 64-bit value in _M right the amount specified by _Count
	//while shifting in zeros
	__m64 _m_psrlqi(__m64 _M, int _Count);//_mm_srli_si64

	/* Logical intrinsics */
	//Performs a bitwise AND of the 64-bit value in _MM1 with the 64-bit value in _MM2
	__m64 _m_pand(__m64 _MM1, __m64 _MM2);//_mm_and_si64
	//Performs a logical NOT on the 64-bit value in _MM1 and use the result in a
	//bitwise AND with the 64-bit value in _MM2
	__m64 _m_pandn(__m64 _MM1, __m64 _MM2);//_mm_andnot_si64
	//Performs a bitwise OR of the 64-bit value in _MM1 with the 64-bit value in _MM2
	__m64 _m_por(__m64 _MM1, __m64 _MM2);//_mm_or_si64
	//Performs a bitwise XOR of the 64-bit value in _MM1 with the 64-bit value in _MM2
	__m64 _m_pxor(__m64 _MM1, __m64 _MM2);//_mm_xor_si64

	/* Comparison intrinsics */
	//If the respective 8-bit values in _MM1 are equal to the respective
	//8-bit values in _MM2, sets the respective 8-bit resulting values to
	//all ones; otherwise, sets them to all zeros
	__m64 _m_pcmpeqb(__m64 _MM1, __m64 _MM2);//_mm_cmpeq_pi8
	//If the respective 16-bit values in _MM1 are equal to the respective
	//16-bit values in _MM2, sets the respective 16-bit resulting values
	//to all ones; otherwise, sets them to all zeros
	__m64 _m_pcmpeqw(__m64 _MM1, __m64 _MM2);//_mm_cmpeq_pi16
	//If the respective 32-bit values in _MM1 are equal to the respective
	//32-bit values in _MM2, sets the respective 32-bit resulting values
	//to all ones; otherwise, sets them to all zeros
	__m64 _m_pcmpeqd(__m64 _MM1, __m64 _MM2);//_mm_cmpeq_pi32
	//If the respective 8-bit values in _MM1 are greater than the respective
	//8-bit values in _MM2, sets the respective 8-bit resulting values to all ones;
	//otherwise, sets them to all zeros
	__m64 _m_pcmpgtb(__m64 _MM1, __m64 _MM2);//_mm_cmpgt_pi8
	//If the respective 16-bit values in _MM1 are greater than the respective 16-bit
	//values in _MM2, sets the respective 16-bit resulting values to all ones;
	//otherwise, sets them to all zeros
	__m64 _m_pcmpgtw(__m64 _MM1, __m64 _MM2);//_mm_cmpgt_pi16
	//If the respective 32-bit values in _MM1 are greater than the respective 32-bit
	//values in _MM2, sets the respective 32-bit resulting values to all ones;
	//otherwise, sets them all to zeros
	__m64 _m_pcmpgtd(__m64 _MM1, __m64 _MM2);//_mm_cmpgt_pi32

	/* Utility intrinsics */
	//Sets the 64-bit value to zero
	__m64 _mm_setzero_si64(void);
	//Sets the two signed 32-bit integer values,r0=_I0, r1=_I1
	__m64 _mm_set_pi32(int _I1, int _I0);
	//r0=_S0, r1=_S1, r2=_S2, r3=_S3
	__m64 _mm_set_pi16(short _S3, short _S2, short _S1, short _S0);
	//r0=_B0, r1=_B1, r2=_B2, r3=_B3, r4=_B4, ..., r7=_B7
	__m64 _mm_set_pi8(char _B7, char _B6, char _B5, char _B4,
		char _B3, char _B2, char _B1, char _B0);
	//Sets the two signed 32-bit integer values to _I,r0=r1=_I
	__m64 _mm_set1_pi32(int _I);
	//Sets the four signed 16-bit integer values to _S, r0=r1=r2=r3=_S
	__m64 _mm_set1_pi16(short _S);
	//Sets the eight signed 8-bit integer values to _B, r0=r1...=r7=_B
	__m64 _mm_set1_pi8(char _B);
	//Sets the two signed 32-bit integer values in reverse order,r0=_I1, r1=_I0
	__m64 _mm_setr_pi32(int _I1, int _I0);
	//Sets the four signed 16-bit integer values in reverse order,
	//r0=_S3, r1=_S2, r2=_S1, r3=_S0
	__m64 _mm_setr_pi16(short _S3, short _S2, short _S1, short _S0);
	//Sets the eight signed 8-bit integer values in reverse order
	//r0=_B7, r1=_B6, r2=_B5, r3=_B4, r4=_B3, r5=_B2, r6=_B1, r7=_B0
	__m64 _mm_setr_pi8(char _B7, char _B6, char _B5, char _B4,
		char _B3, char _B2, char _B1, char _B0);

	/* Alternate intrinsic name definitions */
#define _mm_empty         _m_empty
#define _mm_cvtsi32_si64  _m_from_int
#define _mm_cvtsi64_si32  _m_to_int
#define _mm_packs_pi16    _m_packsswb
#define _mm_packs_pi32    _m_packssdw
#define _mm_packs_pu16    _m_packuswb
#define _mm_unpackhi_pi8  _m_punpckhbw
#define _mm_unpackhi_pi16 _m_punpckhwd
#define _mm_unpackhi_pi32 _m_punpckhdq
#define _mm_unpacklo_pi8  _m_punpcklbw
#define _mm_unpacklo_pi16 _m_punpcklwd
#define _mm_unpacklo_pi32 _m_punpckldq
#define _mm_add_pi8       _m_paddb
#define _mm_add_pi16      _m_paddw
#define _mm_add_pi32      _m_paddd
#define _mm_adds_pi8      _m_paddsb
#define _mm_adds_pi16     _m_paddsw
#define _mm_adds_pu8      _m_paddusb
#define _mm_adds_pu16     _m_paddusw
#define _mm_sub_pi8       _m_psubb
#define _mm_sub_pi16      _m_psubw
#define _mm_sub_pi32      _m_psubd
#define _mm_subs_pi8      _m_psubsb
#define _mm_subs_pi16     _m_psubsw
#define _mm_subs_pu8      _m_psubusb
#define _mm_subs_pu16     _m_psubusw
#define _mm_madd_pi16     _m_pmaddwd
#define _mm_mulhi_pi16    _m_pmulhw
#define _mm_mullo_pi16    _m_pmullw
#define _mm_sll_pi16      _m_psllw
#define _mm_slli_pi16     _m_psllwi
#define _mm_sll_pi32      _m_pslld
#define _mm_slli_pi32     _m_pslldi
#define _mm_sll_si64      _m_psllq
#define _mm_slli_si64     _m_psllqi
#define _mm_sra_pi16      _m_psraw
#define _mm_srai_pi16     _m_psrawi
#define _mm_sra_pi32      _m_psrad
#define _mm_srai_pi32     _m_psradi
#define _mm_srl_pi16      _m_psrlw
#define _mm_srli_pi16     _m_psrlwi
#define _mm_srl_pi32      _m_psrld
#define _mm_srli_pi32     _m_psrldi
#define _mm_srl_si64      _m_psrlq
#define _mm_srli_si64     _m_psrlqi
#define _mm_and_si64      _m_pand
#define _mm_andnot_si64   _m_pandn
#define _mm_or_si64       _m_por
#define _mm_xor_si64      _m_pxor
#define _mm_cmpeq_pi8     _m_pcmpeqb
#define _mm_cmpeq_pi16    _m_pcmpeqw
#define _mm_cmpeq_pi32    _m_pcmpeqd
#define _mm_cmpgt_pi8     _m_pcmpgtb
#define _mm_cmpgt_pi16    _m_pcmpgtw
#define _mm_cmpgt_pi32    _m_pcmpgtd
}
