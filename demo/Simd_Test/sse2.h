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

void SSE2FUN()
{
	/*----Floating-Point Intrinsics Using Streaming SIMD Extension 2 Instructions----*/
	//Arithmetic Operations(Floating Point):add、sub、mul、div、sqrt、min、max
	//返回一个__m128d的寄存器,r0=_A0+_B0, r1=_A1
	extern __m128d _mm_add_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=_A0+_B0, r1=_A1+_B1
	extern __m128d _mm_add_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=_A0-_B0, r1=_A1
	extern __m128d _mm_sub_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=_A0-_B0, r1=_A1-_B1
	extern __m128d _mm_sub_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=_A0*_B0, r1=_A1
	extern __m128d _mm_mul_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=_A0*_B0, r1=_A1*_B1
	extern __m128d _mm_mul_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=sqrt(_B0), r1=_A1
	extern __m128d _mm_sqrt_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=sqrt(_A0), r1=sqrt(_A1)
	extern __m128d _mm_sqrt_pd(__m128d _A);
	//返回一个__m128d的寄存器,r0=_A0/_B0, r1=_A1
	extern __m128d _mm_div_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=_A0/_B0, r1=_A1/_B1
	extern __m128d _mm_div_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=min(_A0,_B0), r1=_A1
	extern __m128d _mm_min_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=min(_A0,_B0), r1=min(_A1,_B1)
	extern __m128d _mm_min_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=max(_A0,_B0), r1=_A1
	extern __m128d _mm_max_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=max(_A0,_B0), r1=max(_A1,_B1)
	extern __m128d _mm_max_pd(__m128d _A, __m128d _B);

	//Logical Operations(Floating Point SSE2 Intrinsics):and、or、xor、 andnot
	//返回一个__m128d的寄存器,r0=_A0 & _B0, r1=_A1 & _B1
	extern __m128d _mm_and_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(~_A0) & _B0, r1=(~_A1) & _B1
	extern __m128d _mm_andnot_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=_A0 | _B0, r1=_A1 | _B1
	extern __m128d _mm_or_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=_A0 ^ _B0, r1=_A1 ^ _B1
	extern __m128d _mm_xor_pd(__m128d _A, __m128d _B);

	//Comparisions:==、<、<=、>、>=、!=
	//返回一个__m128d的寄存器,r0=(_A0 == _B0) ? 0xffffffffffffffff : 0x0, r1=_A1
	extern __m128d _mm_cmpeq_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 == _B0) ? 0xffffffffffffffff : 0x0, 
	//r1=(_A1 == _B1) ? 0xffffffffffffffff : 0x0
	extern __m128d _mm_cmpeq_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 < _B0) ? 0xffffffffffffffff : 0x0, r1=_A1
	extern __m128d _mm_cmplt_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 < _B0) ? 0xffffffffffffffff : 0x0, 
	//r1=(_A1 < _B1) ? 0xffffffffffffffff : 0x0
	extern __m128d _mm_cmplt_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 <= _B0) ? 0xffffffffffffffff : 0x0, r1=_A1
	extern __m128d _mm_cmple_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 <= _B0) ? 0xffffffffffffffff : 0x0, 
	//r1=(_A1 <= _B1) ? 0xffffffffffffffff : 0x0
	extern __m128d _mm_cmple_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 > _B0) ? 0xffffffffffffffff : 0x0, r1=_A1
	extern __m128d _mm_cmpgt_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 > _B0) ? 0xffffffffffffffff : 0x0, 
	//r1=(_A1 > _B1) ? 0xffffffffffffffff : 0x0
	extern __m128d _mm_cmpgt_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 >= _B0) ? 0xffffffffffffffff : 0x0, r1=_A1
	extern __m128d _mm_cmpge_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 >= _B0) ? 0xffffffffffffffff : 0x0, 
	//r1=(_A1 >= _B1) ? 0xffffffffffffffff : 0x0
	extern __m128d _mm_cmpge_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 != _B0) ? 0xffffffffffffffff : 0x0, r1=_A1
	extern __m128d _mm_cmpneq_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 != _B0) ? 0xffffffffffffffff : 0x0, 
	//r1=(_A1 != _B1) ? 0xffffffffffffffff : 0x0
	extern __m128d _mm_cmpneq_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=!(_A0 < _B0) ? 0xffffffffffffffff : 0x0, r1=_A1
	extern __m128d _mm_cmpnlt_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=!(_A0 < _B0) ? 0xffffffffffffffff : 0x0, 
	//r1=!(_A1 < _B1) ? 0xffffffffffffffff : 0x0
	extern __m128d _mm_cmpnlt_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=!(_A0 <= _B0) ? 0xffffffffffffffff : 0x0, r1=_A1
	extern __m128d _mm_cmpnle_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=!(_A0 <= _B0) ? 0xffffffffffffffff : 0x0, 
	//r1=!(_A1 <= _B1) ? 0xffffffffffffffff : 0x0
	extern __m128d _mm_cmpnle_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=!(_A0 > _B0) ? 0xffffffffffffffff : 0x0, r1=_A1
	extern __m128d _mm_cmpngt_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=!(_A0 > _B0) ? 0xffffffffffffffff : 0x0, 
	//r1=!(_A1 > _B1) ? 0xffffffffffffffff : 0x0
	extern __m128d _mm_cmpngt_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=!(_A0 >= _B0) ? 0xffffffffffffffff : 0x0, r1=_A1
	extern __m128d _mm_cmpnge_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=!(_A0 >= _B0) ? 0xffffffffffffffff : 0x0,
	//r1=!(_A1 >= _B1) ? 0xffffffffffffffff : 0x0
	extern __m128d _mm_cmpnge_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 ord _B0) ? 0xffffffffffffffff : 0x0, 
	//r1=(_A1 ord _B1) ? 0xffffffffffffffff : 0x0
	extern __m128d _mm_cmpord_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 ord _B0) ? 0xffffffffffffffff : 0x0, r1=_A1
	extern __m128d _mm_cmpord_sd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 unord _B0) ? 0xffffffffffffffff : 0x0, 
	//r1=(_A1 unord _B1) ? 0xffffffffffffffff : 0x0
	extern __m128d _mm_cmpunord_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(_A0 unord _B0) ? 0xffffffffffffffff : 0x0, r1=_A1
	extern __m128d _mm_cmpunord_sd(__m128d _A, __m128d _B);
	//返回一个0或1的整数,r=(_A0 != _B0) ? 0x1 : 0x0, If _A and _B is a NaN, 1 is returned
	extern int _mm_comieq_sd(__m128d _A, __m128d _B);
	//返回一个0或1的整数,r=(_A0 < _B0) ? 0x1 : 0x0, If _A and _B is a NaN, 1 is returned
	extern int _mm_comilt_sd(__m128d _A, __m128d _B);
	//返回一个0或1的整数,r=(_A0 <= _B0) ? 0x1 : 0x0, If _A and _B is a NaN, 1 is returned
	extern int _mm_comile_sd(__m128d _A, __m128d _B);
	//返回一个0或1的整数,r=(_A0 > _B0) ? 0x1 : 0x0, If _A and _B is a NaN, 0 is returned
	extern int _mm_comigt_sd(__m128d _A, __m128d _B);
	//返回一个0或1的整数,r=(_A0 >= _B0) ? 0x1 : 0x0, If _A and _B is a NaN, 0 is returned
	extern int _mm_comige_sd(__m128d _A, __m128d _B);
	//返回一个0或1的整数,r=(_A0 != _B0) ? 0x1 : 0x0, If _A and _B is a NaN, 0 is returned
	extern int _mm_comineq_sd(__m128d _A, __m128d _B);
	//返回一个0或1的整数,r=(_A0 == _B0) ? 0x1 : 0x0, If _A and _B is a NaN, 1 is returned
	extern int _mm_ucomieq_sd(__m128d _A, __m128d _B);
	//返回一个0或1的整数,r=(_A0 < _B0) ? 0x1 : 0x0, If _A and _B is a NaN, 1 is returned
	extern int _mm_ucomilt_sd(__m128d _A, __m128d _B);
	//返回一个0或1的整数,r=(_A0 <= _B0) ? 0x1 : 0x0, If _A and _B is a NaN, 1 is returned
	extern int _mm_ucomile_sd(__m128d _A, __m128d _B);
	//返回一个0或1的整数,r=(_A0 > _B0) ? 0x1 : 0x0, If _A and _B is a NaN, 0 is returned
	extern int _mm_ucomigt_sd(__m128d _A, __m128d _B);
	//返回一个0或1的整数,r=(_A0 >= _B0) ? 0x1 : 0x0, If _A and _B is a NaN, 0 is returned
	extern int _mm_ucomige_sd(__m128d _A, __m128d _B);
	//返回一个0或1的整数,r=(_A0 != _B0) ? 0x1 : 0x0, If _A and _B is a NaN, 0 is returned
	extern int _mm_ucomineq_sd(__m128d _A, __m128d _B);

	//Conversion Operations
	//返回一个__m128d的寄存器,r0=(dobule)_A0, r1=(double)_A1
	extern __m128d _mm_cvtepi32_pd(__m128i _A);
	//返回一个__m128i的寄存器,r0=(int)_A0, r1=(int)_A1, r2=0x0, r3=0x0
	extern __m128i _mm_cvtpd_epi32(__m128d _A);
	//返回一个__m128i的寄存器,r0=(int)_A0, r1=(int)_A1, r2=0x0, r3=0x0,using truncate
	extern __m128i _mm_cvttpd_epi32(__m128d _A);
	//返回一个__m128的寄存器,r0=(flaot)_A0, r1=(float)_A1, r2=(float)_A2, r3=(float)_A3
	extern __m128 _mm_cvtepi32_ps(__m128i _A);
	//返回一个__m128i的寄存器,r0=(int)_A0, r1=(int)_A1, r2=(int)_A2, r3=(int)_A3
	extern __m128i _mm_cvtps_epi32(__m128 _A);
	//返回一个__m128i的寄存器,r0=(int)_A0, r1=(int)_A1, r2=(int)_A2, r3=(int)_A3,using truncate
	extern __m128i _mm_cvttps_epi32(__m128 _A);
	//返回一个__m128的寄存器,r0=(flaot)_A0, r1=(float)_A1, r2=0.0, r3=0.0
	extern __m128 _mm_cvtpd_ps(__m128d _A);
	//返回一个__m128d的寄存器,r0=(dobule)_A0, r1=(double)_A1
	extern __m128d _mm_cvtps_pd(__m128 _A);
	//返回一个__m128的寄存器,r0=(float)_B0, r1=_B1, r2=_B2, r3=_B3
	extern __m128 _mm_cvtsd_ss(__m128 _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=(double)_B0, r1=_A1
	extern __m128d _mm_cvtss_sd(__m128d _A, __m128 _B);
	//返回一个32bit整数,r=(int)_A0
	extern int _mm_cvtsd_si32(__m128d _A);
	//返回一个32bit整数,r=(int)_A0,using truncate
	extern int _mm_cvttsd_si32(__m128d _A);
	//返回一个__m128d的寄存器,r0=(double)_B, r1=_A1
	extern __m128d _mm_cvtsi32_sd(__m128d _A, int _B);
	//返回一个__m64的寄存器,r0=(int)_A0, r1=(int)_A1
	extern __m64 _mm_cvtpd_pi32(__m128d _A);
	//返回一个__m64的寄存器,r0=(int)_A0, r1=(int)_A1,using truncate
	extern __m64 _mm_cvttpd_pi32(__m128d _A);
	//返回一个__m128d的寄存器,r0=(dobule)_A0, r1=(double)_A1
	extern __m128d _mm_cvtpi32_pd(__m64 _A);

	//Miscellaneous Operations(Floating-Point SSE2 Intrinsics)
	//返回一个__m128d的寄存器,r0=_A1, r1=_B1
	extern __m128d _mm_unpackhi_pd(__m128d _A, __m128d _B);
	//返回一个__m128d的寄存器,r0=_A0, r1=_B0
	extern __m128d _mm_unpacklo_pd(__m128d _A, __m128d _B);
	//返回一个2bit整数,r=sign(_A1) << 1 | sign(_A0)
	extern int _mm_movemask_pd(__m128d _A);
	//返回一个__m128d的寄存器,Selects two specific double-precision,
	// floating-point values from _A and _B, based on the mask _I,
	//The mask must be an immediate
	extern __m128d _mm_shuffle_pd(__m128d _A, __m128d _B, int _I);

	//Load Operations(Floating-Point SSE2 Intrinsics)
	//返回一个__m128d的寄存器,r0=_Dp[0], r1=_Dp[1], The address _Dp must be 16-byte aligned
	extern __m128d _mm_load_pd(double const*_Dp);
	//返回一个__m128d的寄存器,r0=*_Dp, r1=*_Dp, The address _Dp does not need
	//to be 16-byte aligned
	extern __m128d _mm_load1_pd(double const*_Dp);
	//返回一个__m128d的寄存器,r0=_Dp[1], r1=_Dp[0], The address _Dp must be 16-byte aligned
	extern __m128d _mm_loadr_pd(double const*_Dp);
	//返回一个__m128d的寄存器,r0=_Dp[0], r1=_Dp[1], The address _Dp does not 
	//need to be 16-byte aligned
	extern __m128d _mm_loadu_pd(double const*_Dp);
	//返回一个__m128d的寄存器,r0=*_Dp, r1=0.0, The address _Dp does not 
	//need to be 16-byte aligned
	extern __m128d _mm_load_sd(double const*_Dp);
	//返回一个__m128d的寄存器,r0=_A0, r1=*_Dp, The address _Dp does not 
	//need to be 16-byte aligned
	extern __m128d _mm_loadh_pd(__m128d _A, double const*_Dp);
	//返回一个__m128d的寄存器,r0=*_Dp, r1=_A1, The address _Dp does not
	//need to be 16-byte aligned
	extern __m128d _mm_loadl_pd(__m128d _A, double const*_Dp);

	//Set Operations(Floating-Point SSE2 Intrinsics)
	//返回一个__m128d的寄存器,r0=_W, r1=0.0
	extern __m128d _mm_set_sd(double _W);
	//返回一个__m128d的寄存器,r0=_A, r1=_A
	extern __m128d _mm_set1_pd(double _A);
	//返回一个__m128d的寄存器,r0=_Y, r1=_Z
	extern __m128d _mm_set_pd(double _Z, double _Y);
	//返回一个__m128d的寄存器,r0=_Y, r1=_Z
	extern __m128d _mm_setr_pd(double _Y, double _Z);
	//返回一个__m128d的寄存器,r0=0.0, r1=0.0
	extern __m128d _mm_setzero_pd(void);
	//返回一个__m128d的寄存器,r0=_B0, r1=_A1
	extern __m128d _mm_move_sd(__m128d _A, __m128d _B);

	//Store Operations(Floating-Point SSE2 Intrinsics)
	//返回为空,*_Dp=_A0, The address _Dp does not need to be 16-byte aligned
	extern void _mm_store_sd(double *_Dp, __m128d _A);
	//返回为空,_Dp[0]=_A0, _Dp[1]=_A0, The address _Dp must be 16-byte aligned
	extern void _mm_store1_pd(double *_Dp, __m128d _A);
	//返回为空,_Dp[0]=_A0, _Dp[1]=_A1, The address _Dp must be 16-byte aligned
	extern void _mm_store_pd(double *_Dp, __m128d _A);
	//返回为空,_Dp[0]=_A0, _Dp[1]=_A1, The address _Dp does not need to be 16-byte aligned
	extern void _mm_storeu_pd(double *_Dp, __m128d _A);
	//返回为空,_Dp[0]=_A1, _Dp[1]=_A0, The address _Dp must be 16-byte aligned
	extern void _mm_storer_pd(double *_Dp, __m128d _A);
	//返回为空,*_Dp=_A1
	extern void _mm_storeh_pd(double *_Dp, __m128d _A);
	//返回为空,*_Dp=_A0
	extern void _mm_storel_pd(double *_Dp, __m128d _A);

	//new convert to float
	//返回一个64bit double类型,r=_A0, Extracts the lower order floating point value
	extern double _mm_cvtsd_f64(__m128d _A);

	//Cache Support for Streaming SIMD Extensions 2 Floating-Point Operations
	//返回为空,_Dp[0]=_A0, _Dp[1]=_A1, Stores the data in _A to the address _Dp without
	//polluting caches. The address _Dp must be 16-byte aligned. If the cache line 
	//containing address _Dp is already in the cache, the cache will be updated
	extern void _mm_stream_pd(double *_Dp, __m128d _A);

	/*------------Integer Intrinsics Using Streaming SIMD Extensions 2-------------*/
	//Arithmetic Operations(Integer SSE2 Intrinsics):add、sub、mul、avg、min、max
	//返回一个__m128i的寄存器,r0=_A0+_B0, r1=_A1+_B1, ... r15=_A15+_B15
	extern __m128i _mm_add_epi8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,将_A和_B中对应位置的16bit有符号或无符号整数分别相加,
	//即ri=_Ai+_Bi(r0=_A0+_B0, r1=_A1+_B1, ... r7=_A7+_B7)
	extern __m128i _mm_add_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=_A0+_B0, r1=_A1+_B1, r2=_A2+_B2, r3=_A3+_B3
	extern __m128i _mm_add_epi32(__m128i _A, __m128i _B);
	//返回一个__m64的寄存器,r=_A+_B
	extern __m64 _mm_add_si64(__m64 _A, __m64 _B);
	//返回一个__m128i的寄存器,r0=_A0+_B0, r1=_A1+_B1
	extern __m128i _mm_add_epi64(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=SignedSaturate(_A0+_B0), r1=SignedSaturate(_A1+_B1), ... 
	//r15=SignedSaturate(_A15+_B15), saturates
	extern __m128i _mm_adds_epi8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,将_A和_B中对应位置的16bit有符号或无符号整数分别相加,
	//r0=SignedSaturate(_A0+_B0), r1=SignedSaturate(_A1+_B1), ... 
	//r7=SignedSaturate(_A7+_B7), 当计算结果溢出时将其置为边界值(saturates)
	extern __m128i _mm_adds_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=UnsignedSaturate(_A0+_B0), r1=UnsignedSaturate(_A1+_B1), ... 
	//r15=UnsignedSaturate(_A15+_B15), saturates
	extern __m128i _mm_adds_epu8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=UnsignedSaturate(_A0+_B0), r1=UnsignedSaturate(_A1+_B1), ... 
	//r7=UnsignedSaturate(_A7+_B7), saturates
	extern __m128i _mm_adds_epu16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=(_A0+_B0)/2, r1=(_A1+_B1)/2, ... r15=(_A15+_B15)/2, rounds
	extern __m128i _mm_avg_epu8(__m128i _A, __m128i _B); 
	//返回一个__m128i的寄存器,将_A和_B中对应位置的16bit无符号整数取平均,
	//即ri=(_Ai+_Bi)/2(r0=(_A0+_B0)/2, r1=(_A1+_B1)/2, ... r7=(_A7+_B7)/2), rounds
	extern __m128i _mm_avg_epu16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,它含有4个有符号或无符号32bit的整数,
	//分别满足：r0=(_A0*_B0)+(_A1*_B1), r1=(_A2*_B2)+(_A3*_B3), 
	//r2=(_A4*_B4)+(_A5*_B5), r3=(_A6*_B6)+(_A7*_B7)
	extern __m128i _mm_madd_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,取_A和_B中对应位置的16bit有符号或无符号整数的最大值,
	//即ri=max(_Ai,_Bi) (r0=max(_A0,_B1), r1=max(_A1,_B1), ... r7=max(_A7,_B7))
	extern __m128i _mm_max_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=max(_A0,_B1), r1=max(_A1,_B1), ... r15=max(_A15,_B15)
	extern __m128i _mm_max_epu8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,取_A和_B中对应位置的16bit有符号或无符号整数的最小值,
	//即ri=min(_Ai, _Bi)(r0=min(_A0,_B1), r1=min(_A1,_B1), ... r7=min(_A7,_B7))
	extern __m128i _mm_min_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=min(_A0,_B1), r1=min(_A1,_B1), ... r15=min(_A15,_B15)
	extern __m128i _mm_min_epu8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,它含8个有符号或无符号16bit的整数,分别为_A和_B对应位置的16bit
	//有符号或无符号整数相乘结果的高16bit数据,即ri=(_Ai*_Bi)[31:16](r0=(_A0*_B0)[31:16], 
	//r1=(_A1*_B1)[31:16] ... r7=(_A7*_B7)[31:16])
	extern __m128i _mm_mulhi_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=(_A0*_B0)[31:16], r1=(_A1*_B1)[31:16] ... r7=(_A7*_B7)[31:16]
	extern __m128i _mm_mulhi_epu16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,它含8个有符号或无符号16bit的整数,分别为_A和_B对应位置的16bit
	//有符号或无符号整数相乘结果的低16bit数据,即ri=(_Ai*_Bi)[15:0](r0=(_A0*_B0)[15:0], 
	//r1=(_A1*_B1)[15:0] ... r7=(_A7*_B7)[15:0])
	extern __m128i _mm_mullo_epi16(__m128i _A, __m128i _B);
	//返回一个__m64的寄存器,r=_A0*_B0
	extern __m64 _mm_mul_su32(__m64 _A, __m64 _B);
	//返回一个__m128i的寄存器,r0=_A0*_B0, r1=_A2*_B2
	extern __m128i _mm_mul_epu32(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=abs(_A0-_B0) + abs(_A1-_B1) + ... + abs(_A7-_B7), 
	//r1=0x0,r2=0x0, r3=0x0, r4=abs(_A8-_B8) + abs(_A9-_B9) + ... + abs(_A15-_B15), 
	//r5=0x0, r6=0x0, r7=0x0
	extern __m128i _mm_sad_epu8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=_A0-_B0, r1=_A1-_B1, ... r15=_A15-_B15
	extern __m128i _mm_sub_epi8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,将_A和_B中对应位置的16bit有符号或无符号整数分别相减,
	//即ri=_Ai-_Bi(r0=_A0-_B0, r1=_A1-_B1, ... r7=_A7-_B7)
	extern __m128i _mm_sub_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=_A0-_B0, r1=_A1-_B1, r2=_A2-_B2, r3=_A3-_B3
	extern __m128i _mm_sub_epi32(__m128i _A, __m128i _B);
	//返回一个__m64的寄存器,r=_A-_B
	extern __m64 _mm_sub_si64(__m64 _A, __m64 _B);
	//返回一个__m128i的寄存器,r0=_A0-_B0, r1=_A1-_B1
	extern __m128i _mm_sub_epi64(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=SignedSaturate(_A0-_B0), r1=SignedSaturate(_A1-_B1), ... 
	//r15=SignedSaturate(_A15-_B15), saturate
	extern __m128i _mm_subs_epi8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,将_A和_B中对应位置的16bit有符号或无符号整数分别相减,
	//当计算结果溢出时将其置为边界值(saturate), r0=SignedSaturate(_A0-_B0), 
	//r1=SignedSaturate(_A1-_B1), ... r7=SignedSaturate(_A7-_B7)
	extern __m128i _mm_subs_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=UnsignedSaturate(_A0-_B0), r1=UnsignedSaturate(_A1-_B1), ...
	//r15=UnsignedSaturate(_A15-_B15), saturate
	extern __m128i _mm_subs_epu8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=UnsignedSaturate(_A0-_B0), r1=UnsignedSaturate(_A1-_B1), ... 
	//r15=UnsignedSaturate(_A7-_B7), saturate
	extern __m128i _mm_subs_epu16(__m128i _A, __m128i _B);

	//Logical Operations(Integer SSE2 Intrinsics):and、or、xor、andnot
	//返回一个__m128i的寄存器,将寄存器_A和寄存器_B的对应位进行按位与运算, r=_A & _B
	extern __m128i _mm_and_si128(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,将寄存器_A每一位取非,然后和寄存器_B的每一位进行按位与运算,
	//r=(~_A) & _B
	extern __m128i _mm_andnot_si128(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,将寄存器_A和寄存器_B的对应位进行按位或运算, r=_A | _B
	extern __m128i _mm_or_si128(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,将寄存器_A和寄存器_B的对应位进行按位异或运算, r=_A ^ _B
	extern __m128i _mm_xor_si128(__m128i _A, __m128i _B);

	//Shift Operations
	//返回一个__m128i的寄存器,r=_A << (_Imm * 8),  _Imm must be an immediate,  
	//shifting in zeros
	extern __m128i _mm_slli_si128(__m128i _A, int _Imm);
	//返回一个__m128i的寄存器,将寄存器_A中的8个16bit整数按照_Count进行相同的逻辑左移,
	//r0=_A0 << _Count, r1=_A1 << _Count, ... r7=_A7 << count,  shifting in zeros
	extern __m128i _mm_slli_epi16(__m128i _A, int _Count);
	//返回一个__m128i的寄存器,将寄存器_A中的8个16bit整数按照_Count寄存器中对应位置的整数
	//进行逻辑左移, r0=_A0 << _Count, r1=_A1 << _Count, ... r7=_A7 << count,  shifting in zeros
	extern __m128i _mm_sll_epi16(__m128i _A, __m128i _Count);
	//返回一个__m128i的寄存器,r0=_A0 << _Count, r1=_A1 << _Count, r2=_A2 << count, 
	//r3=_A3 << count,  shifting in zeros
	extern __m128i _mm_slli_epi32(__m128i _A, int _Count);
	//返回一个__m128i的寄存器,r0=_A0 << _Count, r1=_A1 << _Count, r2=_A2 << count, 
	//r3=_A3 << count,  shifting in zeros
	extern __m128i _mm_sll_epi32(__m128i _A, __m128i _Count);
	//返回一个__m128i的寄存器,r0=_A0 << _Count, r1=_A1 << _Count,  shifting in zeros
	extern __m128i _mm_slli_epi64(__m128i _A, int _Count);
	//返回一个__m128i的寄存器,r0=_A0 << _Count, r1=_A1 << _Count,  shifting in zeros
	extern __m128i _mm_sll_epi64(__m128i _A, __m128i _Count);
	//返回一个__m128i的寄存器,将寄存器_A中的8个16bit整数按照_Count进行相同的算术右移,
	//r0=_A0 >> _Count, r1=_A1 >> _Count, ... r7=_A7 >> count,  shifting in the sign bit
	extern __m128i _mm_srai_epi16(__m128i _A, int _Count);
	//返回一个__m128i的寄存器,将寄存器_A中的8个16bit整数按照_Count寄存器中对应位置的整数进行
	//算术右移,r0=_A0 >> _Count, r1=_A1 >> _Count, ... r7=_A7 >> count,  shifting in the sign bit
	extern __m128i _mm_sra_epi16(__m128i _A, __m128i _Count);
	//返回一个__m128i的寄存器,r0=_A0 >> _Count, r1=_A1 >> _Count, r3=_A3 >> count, 
	//r4=_A4 >> count,  shifting in the sign bit
	extern __m128i _mm_srai_epi32(__m128i _A, int _Count);
	//返回一个__m128i的寄存器,r0=_A0 >> _Count, r1=_A1 >> _Count, r3=_A3 >> count,
	//r4=_A4 >> count,  shifting in the sign bit
	extern __m128i _mm_sra_epi32(__m128i _A, __m128i _Count);
	//返回一个__m128i的寄存器,r=srl(_A, _Imm * 8),   _Imm must be an immediate,  
	//shifting in zeros
	extern __m128i _mm_srli_si128(__m128i _A, int _Imm);
	//返回一个__m128i的寄存器,将寄存器_A中的8个16bit整数按照_Count进行相同的逻辑右移,
	//移位填充值为0,r0=srl(_A0, _Count), r1=srl(_A1, _Count), ... r7=srl(_A7, _Count), 
	//shifting in zeros
	extern __m128i _mm_srli_epi16(__m128i _A, int _Count);
	//返回一个__m128i的寄存器,将寄存器_A中的8个16bit整数按照_Count寄存器中对应位置的整数
	//进行逻辑右移,移位填充值为0, r0=srl(_A0, _Count), r1=srl(_A1, _Count), ... 
	//r7=srl(_A7, _Count),  shifting in zeros
	extern __m128i _mm_srl_epi16(__m128i _A, __m128i _Count);
	//返回一个__m128i的寄存器,r0=srl(_A0, _Count), r1=srl(_A1, _Count), r2=srl(_A2, _Count),
	//r3=srl(_A3, _Count),  shifting in zeros
	extern __m128i _mm_srli_epi32(__m128i _A, int _Count);
	//返回一个__m128i的寄存器,r0=srl(_A0, _Count), r1=srl(_A1, _Count), r2=srl(_A2, _Count),
	//r3=srl(_A3, _Count),  shifting in zeros
	extern __m128i _mm_srl_epi32(__m128i _A, __m128i _Count);
	//返回一个__m128i的寄存器,r0=srl(_A0, _Count), r1=srl(_A1, _Count), shifting in zeros
	extern __m128i _mm_srli_epi64(__m128i _A, int _Count);
	//返回一个__m128i的寄存器,r0=srl(_A0, _Count), r1=srl(_A1, _Count), shifting in zeros
	extern __m128i _mm_srl_epi64(__m128i _A, __m128i _Count);

	//Comparison Intrinsics(SSE2):==、>、<
	//返回一个__m128i的寄存器,r0=(_A0 == _B0) ? 0xff : 0x00, 
	//r1=(_A1 == _B1) ? 0xff : 0x0, ... r15=(_A15 == _B15) ? 0xff : 0x0
	extern __m128i _mm_cmpeq_epi8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,分别比较寄存器_A和寄存器_B对应位置16bit整数是否相等,若相等,
	//该位置返回0xffff,否则返回0x0,即ri=(_Ai==_Bi)?0xffff:0x0(r0=(_A0 == _B0) ? 0xffff : 0x00, 
	//r1=(_A1 == _B1) ? 0xffff : 0x0, ... r7=(_A7 == _B7) ? 0xffff : 0x0)
	extern __m128i _mm_cmpeq_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=(_A0 == _B0) ? 0xffffffff : 0x00, 
	//r1=(_A1 == _B1) ? 0xffffffff : 0x0,
	//r2=(_A2 == _B2) ? 0xffffffff : 0x0, r3=(_A3 == _B3) ? 0xffffffff : 0x0
	extern __m128i _mm_cmpeq_epi32(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=(_A0 > _B0) ? 0xff : 0x00, r1=(_A1 > _B1) ? 0xff : 0x0, ...
	//r15=(_A15 > _B15) ? 0xff : 0x0
	extern __m128i _mm_cmpgt_epi8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,分别比较寄存器_A的每个16bit整数是否大于寄存器_B对应位置16bit的整数,
	//若大于,该位置返回0xffff,否则返回0x0,
	//即ri=(_Ai>_Bi)?0xffff:0x0(r0=(_A0 > _B0) ? 0xffff : 0x00, 
	//r1=(_A1 > _B1) ? 0xffff : 0x0, ... r7=(_A7 > _B7) ? 0xffff : 0x0)
	extern __m128i _mm_cmpgt_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=(_A0 > _B0) ? 0xffffffff : 0x00,
	//r1=(_A1 > _B1) ? 0xffffffff : 0x0,
	//r2=(_A2 > _B2) ? 0xffffffff : 0x0, r3=(_A3 > _B3) ? 0xffffffff : 0x0
	extern __m128i _mm_cmpgt_epi32(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=(_A0 < _B0) ? 0xff : 0x00, r1=(_A1 < _B1) ? 0xff : 0x0, ... 
	//r15=(_A15 < _B15) ? 0xff : 0x0
	extern __m128i _mm_cmplt_epi8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,分别比较寄存器_A的每个16bit整数是否小于寄存器_B对应位置16bit整数,
	//若小于,该位置返回0xffff,否则返回0x0,
	//即ri=(_Ai<_Bi)?0xffff:0x0(r0=(_A0 < _B0) ? 0xffff : 0x00, 
	//r1=(_A1 < _B1) ? 0xffff : 0x0, ... r7=(_A7 < _B7) ? 0xffff : 0x0)
	extern __m128i _mm_cmplt_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=(_A0 < _B0) ? 0xffffffff : 0x00,
	//r1=(_A1 < _B1) ? 0xffffffff : 0x0, 
	//r2=(_A2 < _B2) ? 0xffffffff : 0x0, r3=(_A3 < _B3) ? 0xffffffff : 0x0
	extern __m128i _mm_cmplt_epi32(__m128i _A, __m128i _B);

	//Conversion Intrinsics: int <-----> __m128i
	//返回一个__m128i的寄存器,r0=_A, r1=0x0, r2=0x0, r3=0x0
	extern __m128i _mm_cvtsi32_si128(int _A);
	//返回一个32bit整数,r=_A0
	extern int _mm_cvtsi128_si32(__m128i _A);

	//Miscellaneous Operations(Integer SSE2 Intrinsics)
	//返回一个__m128i的寄存器,r0=SignedSaturate(_A0), r1=SignedSaturate(_A1), ... 
	//r7=SignedSaturate(_A7), r8=SignedSaturate(_B0), r9=SignedSaturate(_B1), ... 
	//r15=SignedSaturate(_B7),  saturate
	extern __m128i _mm_packs_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=SignedSaturate(_A0), r1=SignedSaturate(_A1), 
	//r2=SignedSaturate(_A2),r3=SignedSaturate(_A3), r4=SignedSaturate(_B0), 
	//r5=SignedSaturate(_B1), r6=SignedSaturate(_B2), r7=SignedSaturate(_B3),  saturate
	extern __m128i _mm_packs_epi32(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=UnsignedSaturate(_A0), r1=UnsignedSaturate(_A1), ... 
	//r7=UnsignedSaturate(_A7),r8=UnsignedSaturate(_B0), r9=UnsignedSaturate(_B1), ... 
	//r15=UnsignedSaturate(_B7),  saturate
	extern __m128i _mm_packus_epi16(__m128i _A, __m128i _B);
	//返回一个16bit整数,根据_Imm从_A中8个16bit数中选取对应编号的数,
	//r=(_Imm == 0) ? _A0 : ((_Imm == 1) ? _A1 : ... (_Imm == 7) ? _A7), 
	//_Imm must be an immediate, zero extends
	extern int _mm_extract_epi16(__m128i _A, int _Imm);
	//返回一个__m128i的寄存器,根据_Imm将_A中8个16bit数中对应编号的数替换为_B,
	//r0=(_Imm == 0) ? _B : _A0; r1=(_Imm == 1) : _B : _A1, ... r7=(_Imm == 7) ? _B : _A7
	extern __m128i _mm_insert_epi16(__m128i _A, int _B, int _Imm);
	//返回一个16bit整数,r=(_A15[7] << 15) | (_A14[7] << 14) ... (_A1[7] << 1) | _A0[7], 
	//zero extends the upper bits
	extern int _mm_movemask_epi8(__m128i _A);
	//返回一个__m128i的寄存器,它是将_A中128bit数据以32bit为单位重新排列得到的,_Imm为有
	//一个四元组,表示重新排列的顺序.当_A中原本存储的整数为16bit时,这条指令将其两两一组
	//进行排列.例如,_A=(_A0,_A1,_A2,_A3,_A4,_A5,_A6,_A7), _Imm=(2,3,0,1),其中_Ai为16bit整数,
	//_A0为低位,返回结果为(_A2,_A3,_A0,_A1,_A6,_A7,_A4,_A5),  _Imm must be an immediate
	extern __m128i _mm_shuffle_epi32(__m128i _A, int _Imm);
	//返回一个__m128i的寄存器,它是将_A中高64bit数据以16bit为单位重新排列得到的,_Imm为一个四元组,
	//表示重新排列的顺序._A中低64bit数据顺序不变.例如,_A=(_A0,_A1,_A2,_A3,_A4,_A5,_A6,_A7), 
	//_Imm=(2,3,0,1),其中_Ai为16bit整数,_A0为低位,返回结果为(_A0,_A1,_A2,_A3,_A5,_A4,_A7,_A6), 
	//_Imm must be an immediate 
	extern __m128i _mm_shufflehi_epi16(__m128i _A, int _Imm);
	//返回一个__m128i的寄存器,它是将_A中低64bit数据以16bit为单位重新排列得到的,_Imm为一个四元组,
	//表示重新排列的顺序._A中高64bit数据顺序不变.例如,_A=(_A0,_A1,_A2,_A3,_A4,_A5,_A6,_A7),
	//_Imm=(2,3,0,1),其中_Ai为16bit整数,_A0为低位,返回结果为(_A1,_A0,_A3,_A2,_A5,_A4,_A7,_A6),   
	//_Imm must be an immediate
	extern __m128i _mm_shufflelo_epi16(__m128i _A, int _Imm);
	//返回一个__m128i的寄存器,r0=_A8, r1=_B8, r2=_A9, r3=_B9, ... r14=_A15, r15=_B15
	extern __m128i _mm_unpackhi_epi8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,它将寄存器_A和寄存器_B的高64bit数以16bit为单位交织在一块.
	//例如,_A=(_A0,_A1,_A2,_A3,_A4,_A5,_A6,_A7),_B=(_B0,_B1,_B2,_B3,_B4,_B5,_B6,_B7),
	//其中_Ai,_Bi为16bit整数,_A0,_B0为低位,返回结果为(_A4,_B4,_A5,_B5,_A6,_B6,_A7,_B7),
	//r0=_A4, r1=_B4, r2=_A5, r3=_B5, r4=_A6, r5=_B6, r6=_A7, r7=_B7
	extern __m128i _mm_unpackhi_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,它将寄存器_A和寄存器_B的高64bit数以32bit为单位交织在一块.
	//例如,_A=(_A0,_A1,_A2,_A3,_A4,_A5,_A6,_A7),_B=(_B0,_B1,_B2,_B3,_B4,_B5,_B6,_B7),
	//其中_Ai,_Bi为16bit整数,_A0,_B0为低位,返回结果为(_A4,_A5,_B4,_B5,_A6,_A7,_B6,_B7),
	//r0=_A2, r1=_B2, r2=_A3, r3=_B3
	extern __m128i _mm_unpackhi_epi32(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,它将寄存器_A和寄存器_B的高64bit数以64bit为单位交织在一块.
	//例如,_A=(_A0,_A1,_A2,_A3,_A4,_A5,_A6,_A7),_B=(_B0,_B1,_B2,_B3,_B4,_B5,_B6,_B7),
	//其中_Ai,_Bi为16bit整数,_A0,_B0为低位,
	//返回结果为(_A4,_A5,_A6,_A7,_B4,_B5,_B6,_B7), r0=_A1, r1=_B1
	extern __m128i _mm_unpackhi_epi64(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,r0=_A0, r1=_B0, r2=_A1, r3=_B1, ... r14=_A7, r15=_B7
	extern __m128i _mm_unpacklo_epi8(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,它将寄存器_A和寄存器_B的低64bit数以16bit为单位交织在一块.
	//例如,_A=(_A0,_A1,_A2,_A3,_A4,_A5,_A6,_A7),_B=(_B0,_B1,_B2,_B3,_B4,_B5,_B6,_B7),
	//其中_Ai,_Bi为16bit整数,_A0,_B0为低位,返回结果为(_A0,_B0,_A1,_B1,_A2,_B2,_A3,_B3),
	//r0=_A0, r1=_B0, r2=_A1, r3=_B1, r4=_A2, r5=_B2, r6=_A3, r7=_B3
	extern __m128i _mm_unpacklo_epi16(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,它将寄存器_A和寄存器_B的低64bit数以32bit为单位交织在一块.
	//例如,_A=(_A0,_A1,_A2,_A3,_A4,_A5,_A6,_A7),_B=(_B0,_B1,_B2,_B3,_B4,_B5,_B6,_B7),
	//其中_Ai,_Bi为16bit整数,_A0,_B0为低位,返回结果为(_A0,_A1,_B0,_B1,_A2,_A3,_B2,_B3),
	//r0=_A0, r1=_B0, r2=_A1, r3=_B1
	extern __m128i _mm_unpacklo_epi32(__m128i _A, __m128i _B);
	//返回一个__m128i的寄存器,它将寄存器_A和寄存器_B的低64bit数以32bit为单位交织在一块.
	//例如,_A=(_A0,_A1,_A2,_A3,_A4,_A5,_A6,_A7),_B=(_B0,_B1,_B2,_B3,_B4,_B5,_B6,_B7),
	//其中_Ai,_Bi为16bit整数,_A0,_B0为低位,返回结果为(_A0,_A1,_A2,_A3,_B0,_B1,_B2,_B3), 
	//ro=_A0, r1=_B0
	extern __m128i _mm_unpacklo_epi64(__m128i _A, __m128i _B);

	//Load Operations(Integer SSE2 Intrinsics)
	//返回为一个__m128i的寄存器,它将_P指向的数据读到指定寄存器中,实际使用时,
	//_P一般是通过类型转换得到的, Address _P must be 16-byte aligned
	extern __m128i _mm_load_si128(__m128i const*_P);
	//返回一个__m128i的寄存器,Loads 128-bit value, Address _P does not need be 16-byte aligned
	extern __m128i _mm_loadu_si128(__m128i const*_P);
	//返回一个__m128i的寄存器,r0=*p[63:0], r1=0x0, zeroing the upper 64 bits of the result
	extern __m128i _mm_loadl_epi64(__m128i const*_P);

	//Set Operations(Integer SSE2 Intrinsics)
	//返回一个__m128i的寄存器,r0=_Q0, r1=_Q1
	extern __m128i _mm_set_epi64(__m64 _Q1, __m64 _Q0);
	//返回一个__m128i的寄存器,r0=_I0, r1=_I1, r2=_I2, r3=_I3
	extern __m128i _mm_set_epi32(int _I3, int _I2, int _I1, int _I0);
	//返回一个__m128i的寄存器,使用8个具体的short型数据来设置寄存器存放数据,
	//r0=_W0, r1=_W1, ... r7=_W7
	extern __m128i _mm_set_epi16(short _W7, short _W6, short _W5, short _W4, 
									short _W3, short _W2, short _W1, short _W0);
	//返回一个__m128i的寄存器,r0=_B0, r1=_B1, ... r15=_B15
	extern __m128i _mm_set_epi8(char _B15, char _B14, char _B13, char _B12, char _B11, 
					char _B10, char _B9,char _B8, char _B7, char _B6, char _B5, char _B4, 
					char _B3, char _B2, char _B1, char _B0);
	//返回一个__m128i的寄存器,r0=_Q, r1=_Q
	extern __m128i _mm_set1_epi64(__m64 _Q);
	//返回一个__m128i的寄存器,r0=_I, r1=_I, r2=_I, r3=_I
	extern __m128i _mm_set1_epi32(int _I);
	//返回一个__m128i的寄存器,r0=_W, r1=_W, ... r7=_W
	extern __m128i _mm_set1_epi16(short _W);
	//返回一个__m128i的寄存器,r0=_B, r1=_B, ... r15=_B
	extern __m128i _mm_set1_epi8(char _B);
	//返回一个__m128i的寄存器,r=_Q
	extern __m128i _mm_setl_epi64(__m128i _Q);
	//返回一个__m128i的寄存器,r0=_Q0, r1=_Q1
	extern __m128i _mm_setr_epi64(__m64 _Q0, __m64 _Q1);
	//返回一个__m128i的寄存器,r0=_I0, r1=_I1, r2=_I2, r3=_I3
	extern __m128i _mm_setr_epi32(int _I0, int _I1, int _I2, int _I3);
	//返回一个__m128i的寄存器,r0=_W0, r1=_W1, ... r7=_W7
	extern __m128i _mm_setr_epi16(short _W0, short _W1, short _W2, short _W3, 
									short _W4, short _W5, short _W6, short _W7);
	//返回一个__m128i的寄存器,r0=_B15, r1=_B14, ... r15=_B0
	extern __m128i _mm_setr_epi8(char _B15, char _B14, char _B13, char _B12, char _B11, 
		char _B10, char _B9, char _B8, char _B7, char _B6, char _B5, char _B4,  
		char _B3, char _B2, char _B1, char _B0);
	//返回一个__m128i的寄存器,r=0x0
	extern __m128i _mm_setzero_si128(void);

	//Store Operations(Integer SSE2 Intrinsics)
	//返回为空,它将寄存器_B中的数据存储到_P指向的地址中,实际使用时,
	//_P一般是通过类型转换得到的, *_P = _B, Address _P must be 16-byte aligned
	extern void _mm_store_si128(__m128i *_P, __m128i _B);
	//返回为空,*_P=_B, Address _P does not need to be 16-byte aligned
	extern void _mm_storeu_si128(__m128i *_P, __m128i _B);
	//返回为空,*_P[63:0] =_Q0, lower 64 bits
	extern void _mm_storel_epi64(__m128i *_P, __m128i _Q);
	//返回为空,if(_N0[7]) _P[0]=_D0, if(_N1[7]) _P[1]=_D1, ... if(_N15[7]) _P[15]=_D15, 
	//The high bit of each byte in the selector _N determines whether the corresponding byte 
	//in _D will be stored. Address _P does not need to be 16-byte aligned
	extern void _mm_maskmoveu_si128(__m128i _D, __m128i _N, char *_P);

	//Integer, moves
	//返回一个__m128i的寄存器,r0=_Q0, r1=0x0, zeroing the upper bits
	extern __m128i _mm_move_epi64(__m128i _Q);
	//返回一个__m128i的寄存器,r0=_Q, r1=0x0, zeroing the upper bits
	extern __m128i _mm_movpi64_epi64(__m64 _Q);
	//返回一个__m64的寄存器,r=_Q0
	extern __m64 _mm_movepi64_pi64(__m128i _Q);

	//Cache Support for Steaming SIMD Extensions 2 Integer Operations
	//返回为空,*_P=_A, Stores the data in _A to the address _P without polluting the caches.
	//If the cache line containing address _P is already in the cache, the cache will be updated. 
	//Address _P must be 16-byte aligned
	extern void _mm_stream_si128(__m128i *_P, __m128i _A);
	//返回为空,Cache line containing _P is flushed and invalidated from 
	//all caches in the coherency domain
	extern void _mm_clflush(void const*_P);
	//返回为空,Guarantees that every load instruction that precedes, in program order, the load 
	//fence instruction is globally visible before any load instruction 
	//that follows the fence in program order
	extern void _mm_lfence(void);
	//返回为空,Guarantees that every memory access that precedes, in program order, 
	//the memory fence instruction is globally visible before any memory instruction 
	//that follows the fence in program order
	extern void _mm_mfence(void);
	//返回为空,*_P=_I, Stores the data in _I to the address _P without polluting the caches. 
	//If the cache line containing address _P is already in the cache, the cache will be updated
	extern void _mm_stream_si32(int *_P, int _I);
	//返回为空,The execution of the next instruction is delayed an implementation specific 
	//amount of time. The instruction does not modify the architectural state. This intrinsic
	//provides especially significant performance gain
	extern void _mm_pause(void);

	/*---Support for casting between various SP, DP, INT vector types. Note that these do no 
		conversion of values, they just change the type----*/
	//返回一个__m128的寄存器,Applies a type cast to reinterpret two 64-bit floating 
	//point values passed in as a 128-bit parameter as packed 32-bit floating point values
	extern __m128  _mm_castpd_ps(__m128d);
	//返回一个__m128i的寄存器,Applies a type cast to reinterpret two 64-bit
	//floating point values passed in as a 128-bit parameter as packed 32-bit integers
	extern __m128i _mm_castpd_si128(__m128d);
	//返回一个__m128d的寄存器,Applies a type cast to reinterpret four 32-bit floating 
	//point values passed in as a 128-bit parameter as packed 64-bit floating point values
	extern __m128d _mm_castps_pd(__m128);
	//返回一个__m128i的寄存器,Applies a type cast to reinterpret four 32-bit floating 
	//point values passed in as a 128-bit parameter as packed 32-bit integers
	extern __m128i _mm_castps_si128(__m128);
	//返回一个__m128的寄存器,Applies a type cast to reinterpret four 32-bit integers 
	//passed in as a 128-bit parameter as packed 32-bit floating point values
	extern __m128  _mm_castsi128_ps(__m128i);
	//返回一个__m128d的寄存器,Applies a type cast to reinterpret four 32-bit 
	//integers passed in as a 128-bit parameter as packed 64-bit floating point values
	extern __m128d _mm_castsi128_pd(__m128i);
}
