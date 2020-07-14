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

void AESFUN()
{
	/*
	* Performs 1 round of AES decryption of the first m128i using 
	* the second m128i as a round key. 
	*/
	//The decrypted data. This instruction decrypts data by using an Equivalent Inverse
	//Cipher with a 128 bit key. AES decryption requires 10 iterations of decryption by
	//using a cipher key that is 128 bits. Each iteration uses this instruction, except
	//for the last iteration.The last iteration must be performed by _mm_aesdeclast_si128.
	extern __m128i _mm_aesdec_si128(__m128i v, __m128i rkey);

	/*
	* Performs the last round of AES decryption of the first m128i 
	* using the second m128i as a round key.
	*/
	//The decrypted data for v. This instruction decrypts data by using an Equivalent 
	//Inverse Cipher with a 128 bit key. AES decryption requires 10 iterations of decryption
	//and uses a cipher key that consists of 128 bits. The final iteration must be performed
	//by this instruction. The previous nine iterations use _mm_aesdec_si128.
	extern __m128i _mm_aesdeclast_si128(__m128i v, __m128i rkey);

	/*
	* Performs 1 round of AES encryption of the first m128i using 
	* the second m128i as a round key.
	*/
	//The encrypted form of the data in v. This instruction encrypts data by using an
	//Equivalent Inverse Cipher with a 128 bit key. AES encryption requires 10 
	//iterations of encryption by using a cipher key that is 128 bits. Each iteration 
	//uses this instruction, except for the last iteration. The last iteration must 
	//be performed by _mm_aesenclast_si128.
	extern __m128i _mm_aesenc_si128(__m128i v, __m128i rkey);

	/*
	* Performs the last round of AES encryption of the first m128i
	* using the second m128i as a round key.
	*/
	//The encrypted form of the data in v. This instruction encrypts data by using an 
	//Equivalent Inverse Cipher with a 128 bit key. AES encryption requires 10 iterations
	//of encryption by using a cipher key that is 128 bits. You must perform the final 
	//iteration with this instruction. The previous nine iterations use _mm_aesenc_si128.
	extern __m128i _mm_aesenclast_si128(__m128i v, __m128i rkey);

	/*
	* Performs the InverseMixColumn operation on the source m128i 
	* and stores the result into m128i destination.
	*/
	//The inverted data. To perform decryption, you should use the aesimc instruction on 
	//all the AES expanded round keys. This prepares them for decryption by using the 
	//Equivalent Inverse Cipher.
	extern __m128i _mm_aesimc_si128(__m128i v);

	/*
	* Generates a m128i round key for the input m128i 
	* AES cipher key and byte round constant. 
	* The second parameter must be a compile time constant.
	*/
	//The AES encryption key. AES encryption requires 10 iterations of encryption with 
	//a 128 bit round key. Each round of encryption requires a different key. This 
	//instruction helps generate the round keys. The round keys can be generated 
	//independently of the encryption phase.
	extern __m128i _mm_aeskeygenassist_si128(__m128i ckey, const int rcon);

	/* 
	* Performs carry-less integer multiplication of 64-bit halves 
	* of 128-bit input operands. 
	* The third parameter inducates which 64-bit haves of the input parameters 
	* v1 and v2 should be used. It must be a compile time constant.
	*/
	//The product calculated by multiplying 64 bits of v1 and 64 bits of v2.
	// This instruction performs a multiplication of two 64-bit integers.
	//The multiplication does not calculate a carry bit.详见参考文献
	extern __m128i _mm_clmulepi64_si128(__m128i v1, __m128i v2, const int imm8);
}
