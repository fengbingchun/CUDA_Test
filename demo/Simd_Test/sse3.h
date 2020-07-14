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

void SSE3FUN()
{
	/*New Single precision vector instructions*/
	//a=(a0, a1, a2, a3), b=(b0, b1, b2, b3)
	//则r0=a0-b0, r1=a1+b1, r2=a2-b2, r3=a3+b3
	extern __m128 _mm_addsub_ps(__m128 a, __m128 b);
	//a=(a0, a1, a2, a3), b=(b0, b1, b2, b3)
	//则r0=a0+a1, r1=a2+a3, r2=b0+b1, r3=b2+b3
	extern __m128 _mm_hadd_ps(__m128 a, __m128 b);
	//a=(a0, a1, a2, a3), b=(b0, b1, b2, b3)
	//则r0=a0-a1, r1=a2-a3, r2=b0-b1, r3=b2-b3
	extern __m128 _mm_hsub_ps(__m128 a, __m128 b);
	//a=(a0, a1, a2, a3), 则r0=a1, r1=a1, r2=a3, r3=a3
	extern __m128 _mm_movehdup_ps(__m128 a);
	//a=(a0, a1, a2, a3), 则r0=a0, r1=a0, r2=a2, r3=a2
	extern __m128 _mm_moveldup_ps(__m128 a);

	/*New double precision vector instructions*/
	//a=(a0, a1), b=(b0, b1), 则r0=a0-b0, r1=a1+b1
	extern __m128d _mm_addsub_pd(__m128d a, __m128d b);
	//a=(a0, a1), b=(b0, b1), 则r0=a0+a1, r1=b0+b1
	extern __m128d _mm_hadd_pd(__m128d a, __m128d b);
	//a=(a0, a1), b=(b0, b1), 则r0=a0-a1, r1=b0-b1
	extern __m128d _mm_hsub_pd(__m128d a, __m128d b);
	//r0=r1=dp[0]
	extern __m128d _mm_loaddup_pd(double const * dp);
	//a=(a0, a1),则r0=r1=a0
	extern __m128d _mm_movedup_pd(__m128d a);

	/*New unaligned integer vector load instruction*/
	//load unaligned data using _mm_lddqu_si128 for best performance
	//If the address is not 16-byte aligned, the load begins at the 
	//highest 16-byte-aligned address less than the address of Data
	extern __m128i _mm_lddqu_si128(__m128i const *p);

	/*Miscellaneous new instructions,
	For _mm_monitor p goes in eax, extensions goes in ecx, hints goes in edx*/
	//The monitor instruction sets up an address range for hardware monitoring.
	//The values of extensions and hints correspond to the values in ECX and EDX
	//used by the monitor instruction. They are reserved for future use and should
	//be zero for the SSE3-enabled processor. For more information, 
	//see the Intel or AMD documentation as appropriate.
	extern void _mm_monitor(void const *p, unsigned extensions, unsigned hints);

	/*Miscellaneous new instructions,
	For _mm_mwait, extensions goes in ecx, hints goes in eax*/
	//The mwait instruction instructs the processor to enter a wait state in which the
	//processor is instructed to monitor the address range between extensions and hints
	//and wait for an event or a store to that address range. The values of extensions 
	//and hints are loaded into the ECX and EAX registers. For more information,
	//see the Intel or AMD documentation as appropriate.
	extern void _mm_mwait(unsigned extensions, unsigned hints);
}
