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

void SSEFUN()
{
	/*----------Floating Point Intrinsics Using Streaming SIMD Extensions------------*/
	//Arithmetic Operations(Floating Point ):add、sub、mul、div、sqrt、rcp、min、max
	//---------------------说明：_ps结尾的指令表示对4个单精度浮点数同时进行运算,
	//_ss结尾的指令表示仅对4个单精度浮点数最低位的浮点数进行运算---------------------
	//返回一个__m128的寄存器,仅将寄存器_A和寄存器_B最低对应位置的32bit单精度浮点数相加,
	//其余位置取寄存器_A中的数据,例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器为r=(_A0+_B0, _A1, _A2, _A3)
	extern __m128 _mm_add_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,将寄存器_A和_B的对应位置的32bit单精度浮点数相加,
	//例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器r0=_A0+_B0, r1=_A1+_B1, r2=_A2+_B2, r3=_A3+_B3
	extern __m128 _mm_add_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,仅将寄存器_A和寄存器_B最低对应位置的32bit单精度浮点数相减,
	//其余位置取寄存器_A中的数据,例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器为r=(_A0-_B0, _A1, _A2, _A3)
	extern __m128 _mm_sub_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,将寄存器_A和_B的对应位置的32bit单精度浮点数相减,
	//例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器r0=_A0-_B0, r1=_A1-_B1, r2=_A2-_B2, r3=_A3-_B3
	extern __m128 _mm_sub_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,仅将寄存器_A和寄存器_B最低对应位置的32bit单精度浮点数相乘,
	//其余位置取寄存器_A中的数据,例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器为r=(_A0*_B0, _A1, _A2, _A3)
	extern __m128 _mm_mul_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,将寄存器_A和_B的对应位置的32bit单精度浮点数相乘,
	//例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器r0=_A0*_B0, r1=_A1*_B1, r2=_A2*_B2, r3=_A3*_B3
	extern __m128 _mm_mul_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,仅将寄存器_A和寄存器_B最低对应位置的32bit单精度浮点数相除,
	//其余位置取寄存器_A中的数据,例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器为r=(_A0/_B0, _A1, _A2, _A3)
	extern __m128 _mm_div_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,将寄存器_A和_B的对应位置的32bit单精度浮点数相除,
	//例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器r0=_A0/_B0, r1=_A1/_B1, r2=_A2/_B2, r3=_A3/_B3
	extern __m128 _mm_div_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,仅将寄存器_A最低对应位置的32bit单精度浮点数开平方,
	//其余位置取寄存器_A中的数据,例如_A=(_A0,_A1,_A2,_A3)
	//则返回寄存器为r=(sqrt(_A0), _A1, _A2, _A3)
	extern __m128 _mm_sqrt_ss(__m128 _A);
	//返回一个__m128的寄存器,将寄存器_A中4个32bit单精度浮点数开平方,
	//例如_A=(_A0,_A1,_A2,_A3),则返回寄存器为
	//r=(sqrt(_A0), sqrt(_A1), sqrt(_A2), sqrt(_A3))
	extern __m128 _mm_sqrt_ps(__m128 _A);
	//返回一个__m128的寄存器,仅将寄存器_A最低对应位置的32bit单精度浮点数取倒数,
	//其余位置取寄存器_A中的数据,例如_A=(_A0,_A1,_A2,_A3)
	//则返回寄存器为r=(recip(_A0), _A1, _A2, _A3)
	extern __m128 _mm_rcp_ss(__m128 _A);
	//返回一个__m128的寄存器,将寄存器_A中4个32bit单精度浮点数取倒数,
	//例如_A=(_A0,_A1,_A2,_A3),则返回寄存器为
	//r=(recip(_A0), recip(_A1), recip(_A2), recip(_A3))
	extern __m128 _mm_rcp_ps(__m128 _A);
	//返回一个__m128的寄存器,仅将寄存器_A最低对应位置的32bit单精度浮点数取平方根的倒数,
	//其余位置取寄存器_A中的数据,例如_A=(_A0,_A1,_A2,_A3)
	//则返回寄存器为r=(recip(sqrt(_A0)), _A1, _A2, _A3)
	extern __m128 _mm_rsqrt_ss(__m128 _A);
	//返回一个__m128的寄存器,将寄存器_A中4个32bit单精度浮点数取平方根的倒数,
	//例如_A=(_A0,_A1,_A2,_A3),则返回寄存器为
	//r=(recip(sqrt(_A0)), recip(sqrt(_A1)), recip(sqrt(_A2)), recip(sqrt(_A3)))
	extern __m128 _mm_rsqrt_ps(__m128 _A);
	//返回一个__m128的寄存器,仅将寄存器_A和寄存器_B最低对应位置的32bit单精度浮点数取最小值,
	//其余位置取寄存器_A中的数据,例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器为r=(min(_A0,_B0), _A1, _A2, _A3)
	extern __m128 _mm_min_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,将寄存器_A和_B的对应位置的32bit单精度浮点数取最小值,
	//例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器r0=min(_A0,_B0), r1=min(_A1,_B1), r2=min(_A2,_B2), r3=min(_A3,_B3)
	extern __m128 _mm_min_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,仅将寄存器_A和寄存器_B最低对应位置的32bit单精度浮点数取最大值,
	//其余位置取寄存器_A中的数据,例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器为r=(max(_A0,_B0), _A1, _A2, _A3)
	extern __m128 _mm_max_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,将寄存器_A和_B的对应位置的32bit单精度浮点数取最大值,
	//例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器r0=max(_A0,_B0), r1=max(_A1,_B1), r2=max(_A2,_B2), r3=max(_A3,_B3)
	extern __m128 _mm_max_ps(__m128 _A, __m128 _B);

	//Logical Operations(SSE)：and、andnot、or、xor
	//返回一个__m128的寄存器,将寄存器_A和_B的对应位置的32bit单精度浮点数分别进行按位与运算,
	//例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器r0=_A0 & _B0, r1=_A1 & _B1, r2=_A2 & _B2, r3=_A3 & _B3
	extern __m128 _mm_and_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,将寄存器_A对应位置的32bit单精度浮点数的非和寄存器_B对应位置的32bit
	//单精度浮点数分别进行按位与运算,例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器r0=~_A0 & _B0, r1=~_A1 & _B1, r2=~_A2 & _B2, r3=~_A3 & _B3
	extern __m128 _mm_andnot_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,将寄存器_A和_B的对应位置的32bit单精度浮点数分别进行按位或运算,
	//例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器r0=_A0 | _B0, r1=_A1 | _B1, r2=_A2 | _B2, r3=_A3 | _B3
	extern __m128 _mm_or_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,将寄存器_A和_B的对应位置的32bit单精度浮点数分别进行按位异或运算,
	//例如_A=(_A0,_A1,_A2,_A3), _B=(_B0,_B1,_B2,_B3),
	//则返回寄存器r0=_A0 ^ _B0, r1=_A1 ^ _B1, r2=_A2 ^ _B2, r3=_A3 ^ _B3
	extern __m128 _mm_xor_ps(__m128 _A, __m128 _B);

	//Comparison Intrinsics(SSE):==、<、<=、>、>=、!=、不小于、不小于等于、不大于、不大于等于
	//返回一个__m128的寄存器,Compares for equality,
	//r0=(_A0 == _B0) ? 0xffffffff : 0x0, r1=_A1, r2=_A2, r3=_A3
	extern __m128 _mm_cmpeq_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for equality,
	//r0=(_A0 == _B0) ? 0xffffffff : 0x0, r1=(_A1 == _B1) ? 0xffffffff : 0x0, 
	//r2=(_A2 == _B2) ? 0xffffffff : 0x0, r3=(_A3 == _B3) ? 0xffffffff : 0x0
	extern __m128 _mm_cmpeq_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for less than,
	//r0=(_A0 < _B0) ? 0xffffffff : 0x0, r1=_A1, r2=_A2, r3=_A3
	extern __m128 _mm_cmplt_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for less than,
	//r0=(_A0 < _B0) ? 0xffffffff : 0x0, r1=(_A1 < _B1) ? 0xffffffff : 0x0, 
	//r2=(_A2 < _B2) ? 0xffffffff : 0x0, r3=(_A3 < _B3) ? 0xffffffff : 0x0
	extern __m128 _mm_cmplt_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for less than or equal,
	//r0=(_A0 <= _B0) ? 0xffffffff : 0x0, r1=_A1, r2=_A2, r3=_A3
	extern __m128 _mm_cmple_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for less than or equal,
	//r0=(_A0 <= _B0) ? 0xffffffff : 0x0, r1=(_A1 <= _B1) ? 0xffffffff : 0x0, 
	//r2=(_A2 <= _B2) ? 0xffffffff : 0x0, r3=(_A3 <= _B3) ? 0xffffffff : 0x0
	extern __m128 _mm_cmple_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for greater than,
	//r0=(_A0 > _B0) ? 0xffffffff : 0x0, r1=_A1, r2=_A2, r3=_A3
	extern __m128 _mm_cmpgt_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for greater than,
	//r0=(_A0 > _B0) ? 0xffffffff : 0x0, r1=(_A1 > _B1) ? 0xffffffff : 0x0, 
	//r2=(_A2 > _B2) ? 0xffffffff : 0x0, r3=(_A3 > _B3) ? 0xffffffff : 0x0
	extern __m128 _mm_cmpgt_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for greater than or equal,
	//r0=(_A0 >= _B0) ? 0xffffffff : 0x0, r1=_A1, r2=_A2, r3=_A3
	extern __m128 _mm_cmpge_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for greater than or equal,
	//r0=(_A0 >= _B0) ? 0xffffffff : 0x0, r1=(_A1 >= _B1) ? 0xffffffff : 0x0, 
	//r2=(_A2 >= _B2) ? 0xffffffff : 0x0, r3=(_A3 >= _B3) ? 0xffffffff : 0x0
	extern __m128 _mm_cmpge_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for inequality,
	//r0=(_A0 != _B0) ? 0xffffffff : 0x0, r1=_A1, r2=_A2, r3=_A3
	extern __m128 _mm_cmpneq_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for inequality,
	//r0=(_A0 != _B0) ? 0xffffffff : 0x0, r1=(_A1 != _B1) ? 0xffffffff : 0x0, 
	//r2=(_A2 != _B2) ? 0xffffffff : 0x0, r3=(_A3 != _B3) ? 0xffffffff : 0x0
	extern __m128 _mm_cmpneq_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for not less than,
	//r0= !(_A0 < _B0) ? 0xffffffff : 0x0, r1=_A1, r2=_A2, r3=_A3
	extern __m128 _mm_cmpnlt_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for not less than,
	//r0=!(_A0 < _B0) ? 0xffffffff : 0x0, r1=!(_A1 < _B1) ? 0xffffffff : 0x0, 
	//r2=!(_A2 < _B2) ? 0xffffffff : 0x0, r3=!(_A3 < _B3) ? 0xffffffff : 0x0
	extern __m128 _mm_cmpnlt_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for not less than or equal
	//r0= !(_A0 <= _B0) ? 0xffffffff : 0x0, r1=_A1, r2=_A2, r3=_A3
	extern __m128 _mm_cmpnle_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for not less than or equal
	//r0=!(_A0 <= _B0) ? 0xffffffff : 0x0, r1=!(_A1 <= _B1) ? 0xffffffff : 0x0, 
	//r2=!(_A2 <= _B2) ? 0xffffffff : 0x0, r3=!(_A3 <= _B3) ? 0xffffffff : 0x0
	extern __m128 _mm_cmpnle_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for not greater than,
	//r0=!(_A0 > _B0) ? 0xffffffff : 0x0, r1=_A1, r2=_A2, r3=_A3
	extern __m128 _mm_cmpngt_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for not greater than,
	//r0=!(_A0 > _B0) ? 0xffffffff : 0x0, r1=!(_A1 > _B1) ? 0xffffffff : 0x0, 
	//r2=!(_A2 > _B2) ? 0xffffffff : 0x0, r3=!(_A3 > _B3) ? 0xffffffff : 0x0
	extern __m128 _mm_cmpngt_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for not greater than or equal,
	//r0=!(_A0 >= _B0) ? 0xffffffff : 0x0, r1=_A1, r2=_A2, r3=_A3
	extern __m128 _mm_cmpnge_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for not greater than or equal,
	//r0=!(_A0 >= _B0) ? 0xffffffff : 0x0, r1=!(_A1 >= _B1) ? 0xffffffff : 0x0, 
	//r2=!(_A2 >= _B2) ? 0xffffffff : 0x0, r3=!(_A3 >= _B3) ? 0xffffffff : 0x0
	extern __m128 _mm_cmpnge_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for ordered,
	//r0=(_A0 ord? _B0) ? 0xffffffff : 0x0, r1=_A1, r2=_A2, r3=_A3
	extern __m128 _mm_cmpord_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for ordered,
	//r0=(_A0 ord? _B0) ? 0xffffffff : 0x0, r1=(_A1 ord? _B1) ? 0xffffffff : 0x0, 
	//r2=(_A2 ord? _B2) ? 0xffffffff : 0x0, r3=(_A3 ord? _B3) ? 0xffffffff : 0x0
	extern __m128 _mm_cmpord_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for unordered,
	//r0=(_A0 unord? _B0) ? 0xffffffff : 0x0, r1=_A1, r2=_A2, r3=_A3	
	extern __m128 _mm_cmpunord_ss(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Compares for unordered,
	//r0=(_A0 unord? _B0) ? 0xffffffff : 0x0, r1=(_A1 unord? _B1) ? 0xffffffff : 0x0, 
	//r2=(_A2 unord? _B2) ? 0xffffffff : 0x0, r3=(_A3 unord? _B3) ? 0xffffffff : 0x0
	extern __m128 _mm_cmpunord_ps(__m128 _A, __m128 _B);
	//返回一个0或1的整数,Compares the lower single-precision, floating-point value of
	//a and b for a equal to b,If a and b are equal, 1 is returned. Otherwise,
	//0 is returned. If a or b is a NaN, 1 is returned
	//r=(_A0 == _B0) ? 0x1 : 0x0
	extern int _mm_comieq_ss(__m128 _A, __m128 _B);
	//返回一个0或1的整数,If a is less than b, 1 is returned. Otherwise, 
	//0 is returned. If a or b is a NaN, 1 is returned,
	//r=(_A0 < _B0) ? 0x1 : 0x0
	extern int _mm_comilt_ss(__m128 _A, __m128 _B);
	//返回一个0或1的整数,If a is less than or equal to b, 1 is returned. 
	//Otherwise, 0 is returned. If a or b is a NaN, 1 is returned,
	//r=(_A0 <= _B0) ? 0x1 : 0x0
	extern int _mm_comile_ss(__m128 _A, __m128 _B);
	//返回一个0或1的整数,If a is greater than b, 1 is returned.
	//Otherwise, 0 is returned. If a or b is a NaN, 1 is returned,
	//r=(_A0 > _B0) ? 0x1 : 0x0
	extern int _mm_comigt_ss(__m128 _A, __m128 _B);
	//返回一个0或1的整数,If a is greater than or equal to b, 1 is returned. 
	//Otherwise, 0 is returned. If a or b is a NaN, 1 is returned,
	//r=(_A0 >= _B0) ? 0x1 : 0x0
	extern int _mm_comige_ss(__m128 _A, __m128 _B);
	//返回一个0或1的整数,If a and b are not equal, 1 is returned. 
	//Otherwise, 0 is returned. If a or b is a NaN, 1 is returned,
	//r=(_A0 != _B0) ? 0x1 : 0x0
	extern int _mm_comineq_ss(__m128 _A, __m128 _B);
	//返回一个0或1的整数,If a and b are equal, 1 is returned. 
	//Otherwise, 0 is returned. If a or b is a NaN, 1 is returned,
	//r=(_A0 == _B0) ? 0x1 : 0x0	
	extern int _mm_ucomieq_ss(__m128 _A, __m128 _B);
	//返回一个0或1的整数,If a is less than b , 1 is returned. 
	//Otherwise, 0 is returned. If a or b is a NaN, 1 is returned,
	//r=(_A0 < _B0) ? 0x1 : 0x0
	extern int _mm_ucomilt_ss(__m128 _A, __m128 _B);
	//返回一个0或1的整数,If a is less than or equal to b, 1 is returned. 
	//Otherwise, 0 is returned. If a or b is a NaN, 1 is returned,
	//r=(_A0 <= _B0) ? 0x1 : 0x0
	extern int _mm_ucomile_ss(__m128 _A, __m128 _B);
	//返回一个0或1的整数,If a is greater than b, 1 is returned. 
	//Otherwise, 0 is returned. If a or b is a NaN, 1 is returned,
	//r=(_A0 > _B0) ? 0x1 : 0x0
	extern int _mm_ucomigt_ss(__m128 _A, __m128 _B);
	//返回一个0或1的整数,If a is greater than or equal to b, 1 is returned.
	//Otherwise, 0 is returned,r=(_A0 >= _B0) ? 0x1 : 0x0
	extern int _mm_ucomige_ss(__m128 _A, __m128 _B);
	//返回一个0或1的整数,If a and b are not equal, 1 is returned. 
	//Otherwise, 0 is returned. If a or b is a NaN, 1 is returned,
	//r=(_A0 != _B0) ? 0x1 : 0x0
	extern int _mm_ucomineq_ss(__m128 _A, __m128 _B);

	//Conversion Operations(SSE)
	//返回一个32bit的整数,Converts the lower single-precision, floating-point value
	//of a to a 32-bit integer according to the current rounding mode, r=(int)_A0
	extern int _mm_cvt_ss2si(__m128 _A);//=_mm_cvtss_si32
	//返回一个__m64寄存器,Converts the two lower single-precision, floating-point 
	//values of a to two 32-bit integers according to the current rounding mode, 
	//returning the integers in packed form, r0=(int)_A0, r1=(int)_A1
	extern __m64 _mm_cvt_ps2pi(__m128 _A);//=_mm_cvtps_pi32
	//返回一个32bit的整数,Converts the lower single-precision, floating-point value
	//of a to a 32-bit integer with truncation, r=(int)_A0
	extern int _mm_cvtt_ss2si(__m128 _A);//=_mm_cvttss_si32
	//返回一个__m64寄存器,Converts the two lower single-precision, floating-point 
	//values of a to two 32-bit integer with truncation, returning the integers 
	//in packed form, r0=(int)_A0, r1=(int)_A1
	extern __m64 _mm_cvtt_ps2pi(__m128 _A);//=_mm_cvttps_pi32
	//返回一个__m128的寄存器,Converts the 32-bit integer value b to an single-precision,
	//floating-point value; the upper three single-precision, floating-point values are
	//passed through from a, r0=(float)_B, r1=_A1, r2=_A2, r3=_A3
	extern __m128 _mm_cvt_si2ss(__m128 _A, int _B);//=_mm_cvtsi32_ss 
	//返回一个__m128的寄存器,Converts the two 32-bit integer values in packed form in b
	//to two single-precision, floating-point values; the upper two single-precision, 
	//floating-point values are passed through from a
	//r0=(float)_B0, r1=(float)_B1, r2=_A2, r3=_A3
	extern __m128 _mm_cvt_pi2ps(__m128 _A, __m64 _B);//=_mm_cvtpi32_ps
	//返回一个__m128的寄存器,Converts the four 16-bit signed integer values in a to 
	//four single-precision, floating-point values
	//r0=(float)_A0, r1=(float)_A1, r2=(float)_A2, r3=(float)_A3
	__inline __m128 _mm_cvtpi16_ps(__m64 _A);
	//返回一个__m128的寄存器,Converts the four 16-bit unsigned integer values in a
	//to four single-precision, floating-point values
	//r0=(float)_A0, r1=(float)_A1, r2=(float)_A2, r3=(float)_A3
	__inline __m128 _mm_cvtpu16_ps(__m64 _A);
	//返回一个__m64的寄存器,Converts the four single-precision, floating-point values
	//in a to four signed 16-bit integer values
	//r0=(short)_A0, r1=(short)_A1, r2=(short)_A2, r3=(short)_A3
	__inline __m64 _mm_cvtps_pi16(__m128 _A);
	//返回一个__m128的寄存器,Converts the lower four 8-bit signed integer values in a 
	//to four single-precision, floating-point values
	//r0=(float)_A0, r1=(float)_A1, r2=(float)_A2, r3=(float)_A3
	__inline __m128 _mm_cvtpi8_ps(__m64 _A);
	//返回一个__m128的寄存器,Converts the lower four 8-bit unsigned integer values in a
	//to four single-precision, floating-point values
	//r0=(float)_A0, r1=(float)_A1, r2=(float)_A2, r3=(float)_A3
	__inline __m128 _mm_cvtpu8_ps(__m64 _A);
	//返回一个__m64的寄存器,Converts the four single-precision, floating-point values 
	//in a to the lower four signed 8-bit integer values of the result
	//r0=(char)_A0, r1=(char)_A1, r2=(char)_A2, r3=(char)_A3
	__inline __m64 _mm_cvtps_pi8(__m128 _A);
	//返回一个__m128的寄存器,Converts the two 32-bit signed integer values in a and the
	//two 32-bit signed integer values in b to four single-precision, floating-point values
	//r0=(float)_A0, r1=(float)_A1, r2=(float)_B0, r3=(float)_B1
	__inline __m128 _mm_cvtpi32x2_ps(__m64 _A, __m64 _B);
	//返回一个32bit浮点数,Extracts the lower order floating point value from the parameter
	//r=_A0
	extern float _mm_cvtss_f32(__m128 _A);

	//Miscellaneous Instructions That Use Streaming SIMD Extensions:
	//返回一个__m128的寄存器,Selects four specific single-precision, floating-point 
	//values from a and b, based on the mask i
	extern __m128 _mm_shuffle_ps(__m128 _A, __m128 _B, unsigned int _Imm8);
	//返回一个__m128的寄存器,Selects and interleaves the upper two single-precision,
	//floating-point values from a and b
	//r0=_A2, r1=_B2, r2=_A3, r3=_B3
	extern __m128 _mm_unpackhi_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Selects and interleaves the lower two single-precision,
	//floating-point values from a and b
	//r0=_A0, r1=_B0, r2=_A1, r3=_B1
	extern __m128 _mm_unpacklo_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Sets the upper two single-precision, floating-point 
	//values with 64 bits of data loaded from the address p; the lower two values
	//are passed through from a
	//r0=_A0, r1=_A1, r2=*_P0, r3=*_P1
	extern __m128 _mm_loadh_pi(__m128 _A, __m64 const* _P);
	//返回一个__m128的寄存器,Moves the upper two single-precision, floating-point
	//values of b to the lower two single-precision, floating-point values of the result
	//r3=_A3, r2=_A2, r1=_B3, r0=_B2
	extern __m128 _mm_movehl_ps(__m128 _A, __m128 _B);
	//返回一个__m128的寄存器,Moves the lower two single-precision, floating-point 
	//values of b to the upper two single-precision, floating-point values of the result
	//r3=_B1, r2=_B0, r1=_A1, r0=_A0
	extern __m128 _mm_movelh_ps(__m128 _A, __m128 _B);
	//返回为空,Stores the upper two single-precision, floating-point values of a 
	//to the address p, *_P0=_A2, *_P1=_A3
	extern void _mm_storeh_pi(__m64 *_P, __m128 _A);
	//返回一个__m128的寄存器,Sets the lower two single-precision, floating-point
	//values with 64 bits of data loaded from the address p; the upper two values
	//are passed through from a
	//r0=*_P0, r1=*_P1, r2=_A2, r3=_A3	
	extern __m128 _mm_loadl_pi(__m128 _A, __m64 const* _P);
	//返回为空,Stores the lower two single-precision, floating-point values of a
	//to the address p, *_P0=_A0, *_P1=_A1
	extern void _mm_storel_pi(__m64 *_P, __m128 _A);
	//返回一个整数,Creates a 4-bit mask from the most significant bits of the
	//four single-precision, floating-point values	
	//r=sign(_A3)<<3 | sign(_A2)<<2 | sign(_A1)<<1 | sign(_A0)
	extern int _mm_movemask_ps(__m128 _A);
	//返回一个无符号整数,Returns the contents of the control register
	extern unsigned int _mm_getcsr(void);
	//返回为空,Sets the control register to the value specified
	extern void _mm_setcsr(unsigned int);

	//Memory and Initialization Using Streaming SIMD Extensions
	//Load Operations(SSE)
	//返回一个__m128的寄存器,Loads an single-precision, floating-point value into
	//the low word and clears the upper three words
	//r0=*_P, r1=0.0, r2=0.0, r3=0.0
	extern __m128 _mm_load_ss(float const* _P);
	//返回一个__m128的寄存器,Loads a single single-precision, floating-point value,
	//copying it into all four words
	//r0=*_P0, r1=*_P1, r2=*_P2, r3=*_P3
	extern __m128 _mm_load_ps1(float const* _P);//=_mm_load1_ps
	//返回一个__m128的寄存器,Loads four single-precision, floating-point values
	//The address must be 16-byte aligned
	//r0=_P[0], r1=_P[1], r2=_P[2], r3=_P[3]
	extern __m128 _mm_load_ps(float const* _P);
	//返回一个__m128的寄存器,Loads four single-precision, floating-point values 
	//in reverse order, The address must be 16-byte aligned
	//r0=_P[3], r1=_P[2], r2=_P[1], r3=_P[0]
	extern __m128 _mm_loadr_ps(float const* _P);
	//返回一个__m128的寄存器,Loads four single-precision, floating-point values
	//The address does not need to be 16-byte aligned
	//r0=_P[0], r1=_P[1], r2=_P[2], r3=_P[3]
	extern __m128 _mm_loadu_ps(float const* _P);

	//Set Operations(SSE)
	//返回一个__m128的寄存器,Sets the low word of an single-precision, 
	//floating-point value to w and clears the upper three words
	//r0=_W, r1=r2=r3=0.0
	extern __m128 _mm_set_ss(float _W);
	//返回一个__m128的寄存器,Sets the four single-precision, floating-point values to w
	//r0=r1=r2=r3=_W
	extern __m128 _mm_set_ps1(float _W);//=_mm_set1_ps
	//返回一个__m128的寄存器,Sets the four single-precision, floating-point values to 
	//the four inputs, r0=_D, r1=_C, r2=_B, r3=_A
	extern __m128 _mm_set_ps(float _A, float _B, float _C, float _D);
	//返回一个__m128的寄存器,Sets the four single-precision, floating-point values to
	//the four inputs in reverse order, r0=_A, r1=_B, r2=_C, r3=_D
	extern __m128 _mm_setr_ps(float _A, float _B, float _C, float _D);
	//返回一个__m128的寄存器,Clears the four single-precision, floating-point values
	//r0=r1=r2=r3=0.0
	extern __m128 _mm_setzero_ps(void);

	//Store Operations(SSE)
	//返回为空,Stores the lower single-precision, floating-point value,*_V=_A0
	extern void _mm_store_ss(float *_V, __m128 _A);
	//返回为空,Stores the lower single-precision, floating-point value across four words
	//_V[0]=_A0, _V[1]=_A0, _V[2]=_A0, _V[3]=_A0
	extern void _mm_store_ps1(float *_V, __m128 _A);//=_mm_store1_ps
	//返回为空,Stores four single-precision, floating-point values
	//The address must be 16-byte aligned
	//_V[0]=_A0, _V[1]=_A1, _V[2]=_A2, _V[3]=_A3
	extern void _mm_store_ps(float *_V, __m128 _A);
	//返回为空,Stores four single-precision, floating-point values in reverse order
	//The address must be 16-byte aligned,
	//_V[0]=_A3, _V[1]=_A2, _V[2]=_A1, _V[3]=_A0
	extern void _mm_storer_ps(float *_V, __m128 _A);
	//返回为空,Stores four single-precision, floating-point values,
	//The address does not need to be 16-byte aligned
	//_V[0]=_A0, _V[1]=_A1, _V[2]=_A2, _V[3]=_A3
	extern void _mm_storeu_ps(float *_V, __m128 _A);
	//返回一个__m128的寄存器,Sets the low word to the single-precision, floating-point
	//value of b,The upper 3 single-precision, floating-point values are passed through 
	//from a, r0=_B0, r1=_A1, r2=_A2, r3=_A3		
	extern __m128 _mm_move_ss(__m128 _A, __m128 _B);

	//Integer Intrinsics Using Streaming SIMD Extensions
	//返回一个16bit整数,Extracts one of the four words of a,
	//The selector n must be an immediate,
	//r=(_Imm == 0) ? _A0 : ((_Imm==1) ? _A1 : ((_Imm==2) ? _A2 : _A3))
	extern int _m_pextrw(__m64 _A, int _Imm);//=_mm_extract_pi16
	//返回一个__m64的寄存器,Inserts word d into one of four words of a,
	//The selector n must be an immediate
	//r0=(_Imm==0)? _D : _A0, r1=(_Imm==1)? _D : _A1,
	//r2=(_Imm==2)? _D : _A2, r3=(_Imm==3)? _D : _A3
	extern __m64 _m_pinsrw(__m64 _A, int _D, int _Imm);//=_mm_insert_pi16
	//返回一个__m64的寄存器,Computes the element-wise maximum of the words in a and b,
	//r0=max(_A0, _B0), r1=max(_A1, _B1), r2=max(_A2, _B2), r3=max(_A3, _B3)
	extern __m64 _m_pmaxsw(__m64 _A, __m64 _B);//=_mm_max_pi16
	//返回一个__m64的寄存器,Computes the element-wise maximum of the unsigned bytes in
	//a and b, r0=max(_A0, _B0), r1=max(_A1, _B1), ... r7=max(_A7, _B7)
	extern __m64 _m_pmaxub(__m64 _A, __m64 _B);//=_mm_max_pu8
	//返回一个__m64的寄存器,Computes the element-wise minimum of the words in a and b
	//r0=min(_A0, _B0), r1=min(_A1, _B1), r2=min(_A2, _B2), r3=min(_A3, _B3)
	extern __m64 _m_pminsw(__m64 _A, __m64 _B);//=_mm_min_pi16
	//返回一个__m64的寄存器,Computes the element-wise minimum of the unsigned bytes
	//in a and b, r0=min(_A0, _B0), r1=min(_A1, _B1), ... r7=min(_A7, _B7)
	extern __m64 _m_pminub(__m64 _A, __m64 _B);//=_mm_min_pu8
	//返回一个整数,Creates an 8-bit mask from the most significant bits of the
	//bytes in a, r=sign(_A7)<<7 | sign(_A6)<<6 | ... | sign(_A0)
	extern int _m_pmovmskb(__m64 _A);//=_mm_movemask_pi8
	//返回一个__m64的寄存器,Multiplies the unsigned words in a and b, returning the
	//upper 16 bits of the 32-bit intermediate results,
	//r0=hiword(_A0, _B0), r1=hiword(_A1, _B1), r2=hiword(_A2, _B2), r3=hiword(_A3, _B3)
	extern __m64 _m_pmulhuw(__m64 _A, __m64 _B);//=_mm_mulhi_pu16
	//返回为空,Conditionally stores byte elements of d to address p,The high bit of 
	//each byte in the selector _B determines whether the corresponding byte in _A 
	//will be stored, if (sign(_B0)) _P[0]=_A0, if (sign(_B1)) _P[1]=_A1, ...
	//if (sign(_B7)) _P[7]=_A7
	extern void _m_maskmovq(__m64 _A, __m64 _B, char * _P);//=_mm_maskmove_si64
	//返回一个__m64的寄存器,Computes the (rounded) averages of the unsigned bytes 
	//in a and b, t=(unsigned short)_A0 + (unsigned short)_B0, r0=(t>>1) | (t & 0x01),
	//..., t=(unsigned short)_A7 + (unsigned short)_B7, r7=(t>>1) | (t & 0x01)	
	extern __m64 _m_pavgb(__m64 _A, __m64 _B);//=_mm_avg_pu8
	//返回一个__m64的寄存器,Computes the (rounded) averages of the unsigned words
	//in a and b, t=(unsigned short)_A0 + (unsigned short)_B0, r0=(t>>1) | (t & 0x01),
	//..., t=(unsigned short)_A4 + (unsigned short)_B4, r7=(t>>1) | (t & 0x01)
	extern __m64 _m_pavgw(__m64 _A, __m64 _B);//=_mm_avg_pu16
	//返回一个__m64的寄存器,Computes the sum of the absolute differences of the unsigned
	//bytes in a and b, returning the value in the lower word
	//The upper three words are cleared
	//r0=abs(_A0-_B0) + ... + abs(_A7-_B7), r1=r2=r3=0
	extern __m64 _m_psadbw(__m64, __m64);//=_mm_sad_pu8
	//返回一个__m64的寄存器,Returns a combination of the four words of a.
	//The selector _Imm must be an immediate
	//r0=word(_Imm & 0x03) of _A, r1=word((_Imm>>2) & 0x03) of _A, 
	//r2=word((_Imm>>4) & 0x03) of _A, r1=word((_Imm>>6) & 0x03) of _A, 
	extern __m64 _m_pshufw(__m64 _A, int _Imm);//=_mm_shuffle_pi16

	//Streaming SIMD Extensions that Support the Cache
	//返回为空,Loads one cache line of data from address p to a location closer
	//to the processor, The value _Sel specifies the type of prefetch operation
	extern void _mm_prefetch(char const*_A, int _Sel);
	//返回为空,Stores the data in a to the address p without polluting the caches
	//This intrinsic requires you to empty the multimedia state for the MMX register
	extern void _mm_stream_pi(__m64 * _P, __m64 _A);
	//返回为空,Stores the data in a to the address p without polluting the caches,
	//The address must be 16-byte aligned
	extern void _mm_stream_ps(float *, __m128 _A);
	//返回为空,Guarantees that every preceding store is globally visible 
	//before any subsequent store
	extern void _mm_sfence(void);

	/* Alternate intrinsic names definition */
#define _mm_cvtss_si32    _mm_cvt_ss2si
#define _mm_cvtps_pi32    _mm_cvt_ps2pi
#define _mm_cvttss_si32   _mm_cvtt_ss2si
#define _mm_cvttps_pi32   _mm_cvtt_ps2pi
#define _mm_cvtsi32_ss    _mm_cvt_si2ss
#define _mm_cvtpi32_ps    _mm_cvt_pi2ps
#define _mm_extract_pi16  _m_pextrw
#define _mm_insert_pi16   _m_pinsrw
#define _mm_max_pi16      _m_pmaxsw
#define _mm_max_pu8       _m_pmaxub
#define _mm_min_pi16      _m_pminsw
#define _mm_min_pu8       _m_pminub
#define _mm_movemask_pi8  _m_pmovmskb
#define _mm_mulhi_pu16    _m_pmulhuw
#define _mm_shuffle_pi16  _m_pshufw
#define _mm_maskmove_si64 _m_maskmovq
#define _mm_avg_pu8       _m_pavgb
#define _mm_avg_pu16      _m_pavgw
#define _mm_sad_pu8       _m_psadbw
#define _mm_set1_ps       _mm_set_ps1
#define _mm_load1_ps      _mm_load_ps1
#define _mm_store1_ps     _mm_store_ps1
}
