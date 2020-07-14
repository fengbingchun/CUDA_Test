void NEONFUN()
{
#ifndef __ARM_NEON__
#error You must enable NEON instructions (e.g. -mfloat-abi=softfp -mfpu=neon) to use arm_neon.h
#endif

	/*(1)、正常指令：生成大小相同且类型通常与操作数向量相同的结果向量；
	(2)、长指令：对双字向量操作数执行运算,生成四字向量的结果.所生成的元素一般是操作数元素宽度的两倍,
	并属于同一类型；
	(3)、宽指令：一个双字向量操作数和一个四字向量操作数执行运算,生成四字向量结果.所生成的元素和第一个
	操作数的元素是第二个操作数元素宽度的两倍；
	(4)、窄指令：四字向量操作数执行运算,并生成双字向量结果,所生成的元素一般是操作数元素宽度的一半；
	(5)、饱和指令：当超过数据类型指定的范围则自动限制在该范围内.*/

	/******************************************************Addition*************************/
	/*--1、Vector add(正常指令): vadd -> ri = ai + bi; r, a, b have equal lane sizes--*/
	int8x8_t vadd_s8 (int8x8_t __a, int8x8_t __b);//_mm_add_epi8
	int16x4_t vadd_s16 (int16x4_t __a, int16x4_t __b);//_mm_add_epi16
	int32x2_t vadd_s32 (int32x2_t __a, int32x2_t __b);//_mm_add_epi32
	int64x1_t vadd_s64 (int64x1_t __a, int64x1_t __b);//_mm_add_epi64
	//_mm_add_ps, SSE, use only low 64 bits
	float32x2_t vadd_f32 (float32x2_t __a, float32x2_t __b);
	uint8x8_t vadd_u8 (uint8x8_t __a, uint8x8_t __b);//_mm_add_epi8
	uint16x4_t vadd_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_add_epi16
	uint32x2_t vadd_u32 (uint32x2_t __a, uint32x2_t __b);//_mm_add_epi32
	uint64x1_t vadd_u64 (uint64x1_t __a, uint64x1_t __b);//_mm_add_epi64
	int8x16_t vaddq_s8 (int8x16_t __a, int8x16_t __b);//_mm_add_epi8
	int16x8_t vaddq_s16 (int16x8_t __a, int16x8_t __b);//_mm_add_epi16
	int32x4_t vaddq_s32 (int32x4_t __a, int32x4_t __b);//_mm_add_epi32
	int64x2_t vaddq_s64 (int64x2_t __a, int64x2_t __b);//_mm_add_epi64
	float32x4_t vaddq_f32 (float32x4_t __a, float32x4_t __b);//_mm_add_ps
	uint8x16_t vaddq_u8 (uint8x16_t __a, uint8x16_t __b);//_mm_add_epi8
	uint16x8_t vaddq_u16 (uint16x8_t __a, uint16x8_t __b);//_mm_add_epi16
	uint32x4_t vaddq_u32 (uint32x4_t __a, uint32x4_t __b);//_mm_add_epi32
	uint64x2_t vaddq_u64 (uint64x2_t __a, uint64x2_t __b);//_mm_add_epi64
	/*--2、Vector long add(长指令): vaddl -> ri = ai + bi; a, b have equal lane sizes,
	result is a 128 bit vector of lanes that are twice the width--*/
	int16x8_t vaddl_s8 (int8x8_t __a, int8x8_t __b);
	int32x4_t vaddl_s16 (int16x4_t __a, int16x4_t __b);
	int64x2_t vaddl_s32 (int32x2_t __a, int32x2_t __b);
	uint16x8_t vaddl_u8 (uint8x8_t __a, uint8x8_t __b);
	uint32x4_t vaddl_u16 (uint16x4_t __a, uint16x4_t __b);
	uint64x2_t vaddl_u32 (uint32x2_t __a, uint32x2_t __b);
	/*--3、Vector wide add(宽指令): vaddw -> ri = ai + bi--*/
	int16x8_t vaddw_s8 (int16x8_t __a, int8x8_t __b);
	int32x4_t vaddw_s16 (int32x4_t __a, int16x4_t __b);
	int64x2_t vaddw_s32 (int64x2_t __a, int32x2_t __b);
	uint16x8_t vaddw_u8 (uint16x8_t __a, uint8x8_t __b);
	uint32x4_t vaddw_u16 (uint32x4_t __a, uint16x4_t __b);
	uint64x2_t vaddw_u32 (uint64x2_t __a, uint32x2_t __b);
	/*--4、Vector halving add: vhadd -> ri = (ai + bi) >> 1;
	shifts each result right one bit, Results are truncated--*/
	int8x8_t vhadd_s8 (int8x8_t __a, int8x8_t __b);
	int16x4_t vhadd_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vhadd_s32 (int32x2_t __a, int32x2_t __b);
	uint8x8_t vhadd_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4_t vhadd_u16 (uint16x4_t __a, uint16x4_t __b);
	uint32x2_t vhadd_u32 (uint32x2_t __a, uint32x2_t __b);
	int8x16_t vhaddq_s8 (int8x16_t __a, int8x16_t __b);
	int16x8_t vhaddq_s16 (int16x8_t __a, int16x8_t __b)
		int32x4_t vhaddq_s32 (int32x4_t __a, int32x4_t __b)
		uint8x16_t vhaddq_u8 (uint8x16_t __a, uint8x16_t __b)
		uint16x8_t vhaddq_u16 (uint16x8_t __a, uint16x8_t __b)
		uint32x4_t vhaddq_u32 (uint32x4_t __a, uint32x4_t __b);
	/*--5、Vector rounding halving add: vrhadd -> ri = (ai + bi + 1) >> 1;
	shifts each result right one bit, Results are rounded(四舍五入)--*/
	int8x8_t vrhadd_s8 (int8x8_t __a, int8x8_t __b);
	int16x4_t vrhadd_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vrhadd_s32 (int32x2_t __a, int32x2_t __b);
	uint8x8_t vrhadd_u8 (uint8x8_t __a, uint8x8_t __b);//_mm_avg_epu8
	uint16x4_t vrhadd_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_avg_epu16
	uint32x2_t vrhadd_u32 (uint32x2_t __a, uint32x2_t __b);
	int8x16_t vrhaddq_s8 (int8x16_t __a, int8x16_t __b);
	int16x8_t vrhaddq_s16 (int16x8_t __a, int16x8_t __b);
	int32x4_t vrhaddq_s32 (int32x4_t __a, int32x4_t __b);
	uint8x16_t vrhaddq_u8 (uint8x16_t __a, uint8x16_t __b);//_mm_avg_epu8
	uint16x8_t vrhaddq_u16 (uint16x8_t __a, uint16x8_t __b);//_mm_avg_epu16
	uint32x4_t vrhaddq_u32 (uint32x4_t __a, uint32x4_t __b);
	/*--6、Vector saturating add(饱和指令): vqadd -> ri = sat(ai + bi);
	the results are saturated if they overflow--*/
	int8x8_t vqadd_s8 (int8x8_t __a, int8x8_t __b);//_mm_adds_epi8
	int16x4_t vqadd_s16 (int16x4_t __a, int16x4_t __b);//_mm_adds_epi16
	int32x2_t vqadd_s32 (int32x2_t __a, int32x2_t __b);
	int64x1_t vqadd_s64 (int64x1_t __a, int64x1_t __b);
	uint8x8_t vqadd_u8 (uint8x8_t __a, uint8x8_t __b);//_mm_adds_epu8
	uint16x4_t vqadd_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_adds_epu16
	uint32x2_t vqadd_u32 (uint32x2_t __a, uint32x2_t __b);
	uint64x1_t vqadd_u64 (uint64x1_t __a, uint64x1_t __b);
	int8x16_t vqaddq_s8 (int8x16_t __a, int8x16_t __b);//_mm_adds_epi8
	int16x8_t vqaddq_s16 (int16x8_t __a, int16x8_t __b);//_mm_adds_epi16
	int32x4_t vqaddq_s32 (int32x4_t __a, int32x4_t __b);
	int64x2_t vqaddq_s64 (int64x2_t __a, int64x2_t __b);
	uint8x16_t vqaddq_u8 (uint8x16_t __a, uint8x16_t __b);//_mm_adds_epu8
	uint16x8_t vqaddq_u16 (uint16x8_t __a, uint16x8_t __b);//_mm_adds_epu16
	uint32x4_t vqaddq_u32 (uint32x4_t __a, uint32x4_t __b);
	uint64x2_t vqaddq_u64 (uint64x2_t __a, uint64x2_t __b);
	/*--7、Vector add high half(窄指令): vaddhn -> ri = sat(ai + bi);
	selecting High half, The results are truncated--*/
	int8x8_t vaddhn_s16 (int16x8_t __a, int16x8_t __b);
	int16x4_t vaddhn_s32 (int32x4_t __a, int32x4_t __b);
	int32x2_t vaddhn_s64 (int64x2_t __a, int64x2_t __b);
	uint8x8_t vaddhn_u16 (uint16x8_t __a, uint16x8_t __b);
	uint16x4_t vaddhn_u32 (uint32x4_t __a, uint32x4_t __b);
	uint32x2_t vaddhn_u64 (uint64x2_t __a, uint64x2_t __b);
	/*--8、Vector rounding add high half(窄指令): vraddhn -> ri = ai + bi;
	selecting High half, The results are rounded--*/
	int8x8_t vraddhn_s16 (int16x8_t __a, int16x8_t __b);
	int16x4_t vraddhn_s32 (int32x4_t __a, int32x4_t __b)
		int32x2_t vraddhn_s64 (int64x2_t __a, int64x2_t __b)
		uint8x8_t vraddhn_u16 (uint16x8_t __a, uint16x8_t __b)
		uint16x4_t vraddhn_u32 (uint32x4_t __a, uint32x4_t __b)
		uint32x2_t vraddhn_u64 (uint64x2_t __a, uint64x2_t __b);
	/*******************************************Multiplication******************************/
	/*--1、Vector multiply(正常指令): vmul -> ri = ai * bi;--*/
	int8x8_t vmul_s8 (int8x8_t __a, int8x8_t __b);
	int16x4_t vmul_s16 (int16x4_t __a, int16x4_t __b);//_mm_mullo_epi16
	int32x2_t vmul_s32 (int32x2_t __a, int32x2_t __b);
	float32x2_t vmul_f32 (float32x2_t __a, float32x2_t __b);//_mm_mul_ps
	uint8x8_t vmul_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4_t vmul_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_mullo_epi16
	uint32x2_t vmul_u32 (uint32x2_t __a, uint32x2_t __b);
	poly8x8_t vmul_p8 (poly8x8_t __a, poly8x8_t __b);
	int8x16_t vmulq_s8 (int8x16_t __a, int8x16_t __b);
	int16x8_t vmulq_s16 (int16x8_t __a, int16x8_t __b);//_mm_mullo_epi16
	int32x4_t vmulq_s32 (int32x4_t __a, int32x4_t __b);
	float32x4_t vmulq_f32 (float32x4_t __a, float32x4_t __b);//_mm_mul_ps
	uint8x16_t vmulq_u8 (uint8x16_t __a, uint8x16_t __b);
	uint16x8_t vmulq_u16 (uint16x8_t __a, uint16x8_t __b);//_mm_mullo_epi16
	uint32x4_t vmulq_u32 (uint32x4_t __a, uint32x4_t __b);
	poly8x16_t vmulq_p8 (poly8x16_t __a, poly8x16_t __b);
	/*--2、Vector multiply accumulate: vmla -> ri = ai + bi * ci; --*/
	int8x8_t vmla_s8 (int8x8_t __a, int8x8_t __b, int8x8_t __c);
	int16x4_t vmla_s16 (int16x4_t __a, int16x4_t __b, int16x4_t __c);
	int32x2_t vmla_s32 (int32x2_t __a, int32x2_t __b, int32x2_t __c);
	float32x2_t vmla_f32 (float32x2_t __a, float32x2_t __b, float32x2_t __c);
	uint8x8_t vmla_u8 (uint8x8_t __a, uint8x8_t __b, uint8x8_t __c);
	uint16x4_t vmla_u16 (uint16x4_t __a, uint16x4_t __b, uint16x4_t __c);
	uint32x2_t vmla_u32 (uint32x2_t __a, uint32x2_t __b, uint32x2_t __c);
	int8x16_t vmlaq_s8 (int8x16_t __a, int8x16_t __b, int8x16_t __c);
	int16x8_t vmlaq_s16 (int16x8_t __a, int16x8_t __b, int16x8_t __c);
	int32x4_t vmlaq_s32 (int32x4_t __a, int32x4_t __b, int32x4_t __c);
	float32x4_t vmlaq_f32 (float32x4_t __a, float32x4_t __b, float32x4_t __c);
	uint8x16_t vmlaq_u8 (uint8x16_t __a, uint8x16_t __b, uint8x16_t __c);
	uint16x8_t vmlaq_u16 (uint16x8_t __a, uint16x8_t __b, uint16x8_t __c);
	uint32x4_t vmlaq_u32 (uint32x4_t __a, uint32x4_t __b, uint32x4_t __c);
	/*--3、Vector multiply accumulate long: vmlal -> ri = ai + bi * ci --*/
	int16x8_t vmlal_s8 (int16x8_t __a, int8x8_t __b, int8x8_t __c);
	int32x4_t vmlal_s16 (int32x4_t __a, int16x4_t __b, int16x4_t __c);
	int64x2_t vmlal_s32 (int64x2_t __a, int32x2_t __b, int32x2_t __c);
	uint16x8_t vmlal_u8 (uint16x8_t __a, uint8x8_t __b, uint8x8_t __c);
	uint32x4_t vmlal_u16 (uint32x4_t __a, uint16x4_t __b, uint16x4_t __c);
	uint64x2_t vmlal_u32 (uint64x2_t __a, uint32x2_t __b, uint32x2_t __c);
	/*--4、Vector multiply subtract: vmls -> ri = ai - bi * ci --*/
	int8x8_t vmls_s8 (int8x8_t __a, int8x8_t __b, int8x8_t __c);
	int16x4_t vmls_s16 (int16x4_t __a, int16x4_t __b, int16x4_t __c);
	int32x2_t vmls_s32 (int32x2_t __a, int32x2_t __b, int32x2_t __c);
	float32x2_t vmls_f32 (float32x2_t __a, float32x2_t __b, float32x2_t __c);
	uint8x8_t vmls_u8 (uint8x8_t __a, uint8x8_t __b, uint8x8_t __c);
	uint16x4_t vmls_u16 (uint16x4_t __a, uint16x4_t __b, uint16x4_t __c);
	uint32x2_t vmls_u32 (uint32x2_t __a, uint32x2_t __b, uint32x2_t __c);
	int8x16_t vmlsq_s8 (int8x16_t __a, int8x16_t __b, int8x16_t __c);
	int16x8_t vmlsq_s16 (int16x8_t __a, int16x8_t __b, int16x8_t __c);
	int32x4_t vmlsq_s32 (int32x4_t __a, int32x4_t __b, int32x4_t __c);
	float32x4_t vmlsq_f32 (float32x4_t __a, float32x4_t __b, float32x4_t __c);
	uint8x16_t vmlsq_u8 (uint8x16_t __a, uint8x16_t __b, uint8x16_t __c);
	uint16x8_t vmlsq_u16 (uint16x8_t __a, uint16x8_t __b, uint16x8_t __c);
	uint32x4_t vmlsq_u32 (uint32x4_t __a, uint32x4_t __b, uint32x4_t __c);
	/*--5、Vector multiply subtract long：vmlsl -> ri = ai - bi * ci --*/
	int16x8_t vmlsl_s8 (int16x8_t __a, int8x8_t __b, int8x8_t __c);
	int32x4_t vmlsl_s16 (int32x4_t __a, int16x4_t __b, int16x4_t __c);
	int64x2_t vmlsl_s32 (int64x2_t __a, int32x2_t __b, int32x2_t __c);
	uint16x8_t vmlsl_u8 (uint16x8_t __a, uint8x8_t __b, uint8x8_t __c);
	uint32x4_t vmlsl_u16 (uint32x4_t __a, uint16x4_t __b, uint16x4_t __c);
	uint64x2_t vmlsl_u32 (uint64x2_t __a, uint32x2_t __b, uint32x2_t __c);
	/*--6、Vector saturating doubling multiply high: vqdmulh -> ri = sat(ai * bi);
	doubles the results and returns only the high half of the truncated results--*/
	int16x4_t vqdmulh_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vqdmulh_s32 (int32x2_t __a, int32x2_t __b);
	int16x8_t vqdmulhq_s16 (int16x8_t __a, int16x8_t __b);
	int32x4_t vqdmulhq_s32 (int32x4_t __a, int32x4_t __b);
	/*--7、Vector saturating rounding doubling multiply high vqrdmulh -> ri = ai * bi:
	doubles the results and returns only the high half of the rounded results.
	The results are saturated if they overflow--*/
	int16x4_t vqrdmulh_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vqrdmulh_s32 (int32x2_t __a, int32x2_t __b);
	int16x8_t vqrdmulhq_s16 (int16x8_t __a, int16x8_t __b);
	int32x4_t vqrdmulhq_s32 (int32x4_t __a, int32x4_t __b);
	/*--8、Vector saturating doubling multiply accumulate long: vqdmlal -> ri = ai + bi * ci;
	multiplies the elements in the second and third vectors, doubles the results and adds the
	results to the values in the first vector. The results are saturated if they overflow--*/
	int32x4_t vqdmlal_s16 (int32x4_t __a, int16x4_t __b, int16x4_t __c);
	int64x2_t  vqdmlal_s32 (int64x2_t __a, int32x2_t __b, int32x2_t __c);
	/*--9、Vector saturating doubling multiply subtract long: vqdmlsl -> ri = ai - bi * ci;
	multiplies the elements in the second and third vectors, doubles the results and subtracts
	the results from the elements in the first vector.
	The results are saturated if they overflow--*/
	int32x4_t vqdmlsl_s16 (int32x4_t __a, int16x4_t __b, int16x4_t __c);
	int64x2_t vqdmlsl_s32 (int64x2_t __a, int32x2_t __b, int32x2_t __c);
	/*--10、Vector long multiply(长指令): vmull -> ri = ai * bi;--*/
	int16x8_t vmull_s8 (int8x8_t __a, int8x8_t __b);
	int32x4_t vmull_s16 (int16x4_t __a, int16x4_t __b);
	int64x2_t vmull_s32 (int32x2_t __a, int32x2_t __b);
	uint16x8_t vmull_u8 (uint8x8_t __a, uint8x8_t __b);
	uint32x4_t vmull_u16 (uint16x4_t __a, uint16x4_t __b);
	uint64x2_t vmull_u32 (uint32x2_t __a, uint32x2_t __b);
	poly16x8_t vmull_p8 (poly8x8_t __a, poly8x8_t __b);
	/*--11、Vector saturating doubling long multiply: vqdmull -> ri = ai * bi;
	If any of the results overflow, they are saturated--*/
	int32x4_t vqdmull_s16 (int16x4_t __a, int16x4_t __b);
	int64x2_t vqdmull_s32 (int32x2_t __a, int32x2_t __b);
	/*--12、Fused multiply accumulate: vfma -> ri = ai + bi * ci;
	The result of the multiply is not rounded before the accumulation--*/
	float32x2_t vfma_f32 (float32x2_t __a, float32x2_t __b, float32x2_t __c)
		float32x4_t vfmaq_f32 (float32x4_t __a, float32x4_t __b, float32x4_t __c);
	/*--13、Fused multiply subtract: vfms -> ri = ai - bi * ci;
	The result of the multiply is not rounded before the subtraction--*/
	float32x2_t vfms_f32 (float32x2_t __a, float32x2_t __b, float32x2_t __c);
	float32x4_t vfmsq_f32 (float32x4_t __a, float32x4_t __b, float32x4_t __c);
	/******************************************************Round to integral****************/
	/*--1、to nearest, ties to even--*/
	float32x2_t vrndn_f32 (float32x2_t __a);
	float32x4_t vrndqn_f32 (float32x4_t __a);
	/*--2、to nearest, ties away from zero--*/
	float32x2_t vrnda_f32 (float32x2_t __a);
	float32x4_t vrndqa_f32 (float32x4_t __a);
	/*--3、towards +Inf--*/
	float32x2_t vrndp_f32 (float32x2_t __a);
	float32x4_t vrndqp_f32 (float32x4_t __a);
	/*--4、towards -Inf--*/
	float32x2_t vrndm_f32 (float32x2_t __a);
	float32x4_t vrndqm_f32 (float32x4_t __a);
	/*--5、towards 0--*/
	float32x2_t vrnd_f32 (float32x2_t __a);
	float32x4_t vrndq_f32 (float32x4_t __a);
	/**********************************************Subtraction******************************/
	/*--1、Vector subtract(正常指令):vsub -> ri = ai - bi;--*/
	int8x8_t vsub_s8 (int8x8_t __a, int8x8_t __b);//_mm_sub_epi8
	int16x4_t vsub_s16 (int16x4_t __a, int16x4_t __b);//_mm_sub_epi16
	int32x2_t vsub_s32 (int32x2_t __a, int32x2_t __b);//_mm_sub_epi32
	int64x1_t vsub_s64 (int64x1_t __a, int64x1_t __b);//_mm_sub_epi64
	float32x2_t vsub_f32 (float32x2_t __a, float32x2_t __b);//_mm_sub_ps
	uint8x8_t vsub_u8 (uint8x8_t __a, uint8x8_t __b);//_mm_sub_epi8
	uint16x4_t vsub_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_sub_epi16
	uint32x2_t vsub_u32 (uint32x2_t __a, uint32x2_t __b);//_mm_sub_epi32
	uint64x1_t vsub_u64 (uint64x1_t __a, uint64x1_t __b);//_mm_sub_epi64
	int8x16_t vsubq_s8 (int8x16_t __a, int8x16_t __b);//_mm_sub_epi8
	int16x8_t vsubq_s16 (int16x8_t __a, int16x8_t __b);//_mm_sub_epi16
	int32x4_t vsubq_s32 (int32x4_t __a, int32x4_t __b);//_mm_sub_epi32
	int64x2_t vsubq_s64 (int64x2_t __a, int64x2_t __b);//_mm_sub_epi64
	float32x4_t vsubq_f32 (float32x4_t __a, float32x4_t __b);//_mm_sub_ps
	uint8x16_t vsubq_u8 (uint8x16_t __a, uint8x16_t __b);//_mm_sub_epi8
	uint16x8_t vsubq_u16 (uint16x8_t __a, uint16x8_t __b);//_mm_sub_epi16
	uint32x4_t vsubq_u32 (uint32x4_t __a, uint32x4_t __b);//_mm_sub_epi32
	uint64x2_t vsubq_u64 (uint64x2_t __a, uint64x2_t __b);//_mm_sub_epi64
	/*--2、Vector long subtract(长指令): vsubl -> ri = ai - bi; --*/
	int16x8_t vsubl_s8 (int8x8_t __a, int8x8_t __b);
	int32x4_t vsubl_s16 (int16x4_t __a, int16x4_t __b);
	int64x2_t vsubl_s32 (int32x2_t __a, int32x2_t __b);
	uint16x8_t vsubl_u8 (uint8x8_t __a, uint8x8_t __b);
	uint32x4_t vsubl_u16 (uint16x4_t __a, uint16x4_t __b);
	uint64x2_t vsubl_u32 (uint32x2_t __a, uint32x2_t __b);
	/*--3、Vector wide subtract(宽指令): vsubw -> ri = ai - bi;--*/
	int16x8_t vsubw_s8 (int16x8_t __a, int8x8_t __b);
	int32x4_t vsubw_s16 (int32x4_t __a, int16x4_t __b);
	int64x2_t vsubw_s32 (int64x2_t __a, int32x2_t __b);
	uint16x8_t vsubw_u8 (uint16x8_t __a, uint8x8_t __b);
	uint32x4_t vsubw_u16 (uint32x4_t __a, uint16x4_t __b);
	uint64x2_t vsubw_u32 (uint64x2_t __a, uint32x2_t __b);
	/*--4、Vector saturating subtract(饱和指令): vqsub -> ri = sat(ai - bi);
	If any of the results overflow, they are saturated--*/
	int8x8_t vqsub_s8 (int8x8_t __a, int8x8_t __b);//_mm_subs_epi8
	int16x4_t vqsub_s16 (int16x4_t __a, int16x4_t __b);//_mm_subs_epi16
	int32x2_t vqsub_s32 (int32x2_t __a, int32x2_t __b);//_mm_subs_epi32
	int64x1_t vqsub_s64 (int64x1_t __a, int64x1_t __b);
	uint8x8_t vqsub_u8 (uint8x8_t __a, uint8x8_t __b);//_mm_subs_epu8
	uint16x4_t vqsub_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_subs_epu16
	uint32x2_t vqsub_u32 (uint32x2_t __a, uint32x2_t __b);//_mm_subs_epu32
	uint64x1_t vqsub_u64 (uint64x1_t __a, uint64x1_t __b);
	int8x16_t vqsubq_s8 (int8x16_t __a, int8x16_t __b);//_mm_subs_epi8
	int16x8_t vqsubq_s16 (int16x8_t __a, int16x8_t __b);//_mm_subs_epi16
	int32x4_t vqsubq_s32 (int32x4_t __a, int32x4_t __b);//_mm_subs_epi32
	int64x2_t vqsubq_s64 (int64x2_t __a, int64x2_t __b);
	uint8x16_t vqsubq_u8 (uint8x16_t __a, uint8x16_t __b);//_mm_subs_epu8
	uint16x8_t vqsubq_u16 (uint16x8_t __a, uint16x8_t __b);//_mm_subs_epu16
	uint32x4_t vqsubq_u32 (uint32x4_t __a, uint32x4_t __b);//_mm_subs_epu32
	uint64x2_t vqsubq_u64 (uint64x2_t __a, uint64x2_t __b);
	/*--5、Vector halving subtract: vhsub -> ri = (ai - bi) >> 1;
	shifts each result right one bit, The results are truncated.--*/
	int8x8_t vhsub_s8 (int8x8_t __a, int8x8_t __b);
	int16x4_t vhsub_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vhsub_s32 (int32x2_t __a, int32x2_t __b);
	uint8x8_t vhsub_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4_t vhsub_u16 (uint16x4_t __a, uint16x4_t __b);
	uint32x2_t vhsub_u32 (uint32x2_t __a, uint32x2_t __b);
	int8x16_t vhsubq_s8 (int8x16_t __a, int8x16_t __b);
	int16x8_t vhsubq_s16 (int16x8_t __a, int16x8_t __b);
	int32x4_t vhsubq_s32 (int32x4_t __a, int32x4_t __b);
	uint8x16_t vhsubq_u8 (uint8x16_t __a, uint8x16_t __b);
	uint16x8_t vhsubq_u16 (uint16x8_t __a, uint16x8_t __b);
	uint32x4_t vhsubq_u32 (uint32x4_t __a, uint32x4_t __b);
	/*--6、Vector subtract high half(窄指令): vsubhn -> ri = ai - bi;
	It returns the most significant halves of the results. The results are truncated--*/
	int8x8_t vsubhn_s16 (int16x8_t __a, int16x8_t __b);
	int16x4_t vsubhn_s32 (int32x4_t __a, int32x4_t __b);
	int32x2_t vsubhn_s64 (int64x2_t __a, int64x2_t __b);
	uint8x8_t vsubhn_u16 (uint16x8_t __a, uint16x8_t __b);
	uint16x4_t vsubhn_u32 (uint32x4_t __a, uint32x4_t __b);
	uint32x2_t vsubhn_u64 (uint64x2_t __a, uint64x2_t __b);
	/*--7、Vector rounding subtract high half(窄指令): vrsubhn -> ai - bi;
	It returns the most significant halves of the results. The results are rounded--*/
	int8x8_t vrsubhn_s16 (int16x8_t __a, int16x8_t __b);
	int16x4_t vrsubhn_s32 (int32x4_t __a, int32x4_t __b);
	int32x2_t vrsubhn_s64 (int64x2_t __a, int64x2_t __b)
		uint8x8_t vrsubhn_u16 (uint16x8_t __a, uint16x8_t __b);
	uint16x4_t vrsubhn_u32 (uint32x4_t __a, uint32x4_t __b);
	uint32x2_t vrsubhn_u64 (uint64x2_t __a, uint64x2_t __b);
	/******************************************************Comparison***********************/
	/*--1、Vector compare equal(正常指令): vceq -> ri = ai == bi ? 1...1 : 0...0;
	If they are equal, the corresponding element in the destination vector is set to all ones.
	Otherwise, it is set to all zeros--*/
	uint8x8_t vceq_s8 (int8x8_t __a, int8x8_t __b);//_mm_cmpeq_epi8
	uint16x4_t vceq_s16 (int16x4_t __a, int16x4_t __b);//_mm_cmpeq_epi16
	uint32x2_t vceq_s32 (int32x2_t __a, int32x2_t __b);//_mm_cmpeq_epi32
	uint32x2_t vceq_f32 (float32x2_t __a, float32x2_t __b);
	uint8x8_t vceq_u8 (uint8x8_t __a, uint8x8_t __b);//_mm_cmpeq_epi8
	uint16x4_t vceq_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_cmpeq_epi16
	uint32x2_t vceq_u32 (uint32x2_t __a, uint32x2_t __b);//_mm_cmpeq_epi32
	uint8x8_t vceq_p8 (poly8x8_t __a, poly8x8_t __b);//_mm_cmpeq_epi8
	uint8x16_t vceqq_s8 (int8x16_t __a, int8x16_t __b);//_mm_cmpeq_epi8
	uint16x8_t vceqq_s16 (int16x8_t __a, int16x8_t __b);//_mm_cmpeq_epi16
	uint32x4_t vceqq_s32 (int32x4_t __a, int32x4_t __b);//_mm_cmpeq_epi32
	uint32x4_t vceqq_f32 (float32x4_t __a, float32x4_t __b);
	uint8x16_t vceqq_u8 (uint8x16_t __a, uint8x16_t __b);//_mm_cmpeq_epi8
	uint16x8_t vceqq_u16 (uint16x8_t __a, uint16x8_t __b);//_mm_cmpeq_epi16
	uint32x4_t vceqq_u32 (uint32x4_t __a, uint32x4_t __b);//_mm_cmpeq_epi32
	uint8x16_t vceqq_p8 (poly8x16_t __a, poly8x16_t __b);//_mm_cmpeq_epi8
	/*--2、Vector compare greater-than or equal(正常指令): vcge-> ri = ai >= bi ? 1...1:0...0;
	If it is greater than or equal to it, the corresponding element in the destination
	vector is set to all ones. Otherwise, it is set to all zeros.--*/
	uint8x8_t vcge_s8 (int8x8_t __a, int8x8_t __b);
	uint16x4_t vcge_s16 (int16x4_t __a, int16x4_t __b);
	uint32x2_t vcge_s32 (int32x2_t __a, int32x2_t __b);
	uint32x2_t vcge_f32 (float32x2_t __a, float32x2_t __b);
	uint8x8_t vcge_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4_t vcge_u16 (uint16x4_t __a, uint16x4_t __b);
	uint32x2_t vcge_u32 (uint32x2_t __a, uint32x2_t __b);
	uint8x16_t vcgeq_s8 (int8x16_t __a, int8x16_t __b);
	uint16x8_t vcgeq_s16 (int16x8_t __a, int16x8_t __b);
	uint32x4_t vcgeq_s32 (int32x4_t __a, int32x4_t __b);
	uint32x4_t vcgeq_f32 (float32x4_t __a, float32x4_t __b);
	uint8x16_t vcgeq_u8 (uint8x16_t __a, uint8x16_t __b);
	uint16x8_t vcgeq_u16 (uint16x8_t __a, uint16x8_t __b);
	uint32x4_t vcgeq_u32 (uint32x4_t __a, uint32x4_t __b);
	/*--3、Vector compare less-than or equal(正常指令): vcle -> ri = ai <= bi ? 1...1:0...0;
	If it is less than or equal to it, the corresponding element in the destination vector
	is set to all ones. Otherwise, it is set to all zeros.--*/
	uint8x8_t vcle_s8 (int8x8_t __a, int8x8_t __b);
	uint16x4_t vcle_s16 (int16x4_t __a, int16x4_t __b);
	uint32x2_t vcle_s32 (int32x2_t __a, int32x2_t __b);
	uint32x2_t vcle_f32 (float32x2_t __a, float32x2_t __b);
	uint8x8_t vcle_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4_t vcle_u16 (uint16x4_t __a, uint16x4_t __b);
	uint32x2_t vcle_u32 (uint32x2_t __a, uint32x2_t __b);
	uint8x16_t vcleq_s8 (int8x16_t __a, int8x16_t __b);
	uint16x8_t vcleq_s16 (int16x8_t __a, int16x8_t __b);
	uint32x4_t vcleq_s32 (int32x4_t __a, int32x4_t __b);
	uint32x4_t vcleq_f32 (float32x4_t __a, float32x4_t __b);
	uint8x16_t vcleq_u8 (uint8x16_t __a, uint8x16_t __b);
	uint16x8_t vcleq_u16 (uint16x8_t __a, uint16x8_t __b);
	uint32x4_t vcleq_u32 (uint32x4_t __a, uint32x4_t __b);
	/*--4、Vector compare greater-than(正常指令): vcgt -> ri = ai > bi ? 1...1:0...0;
	If it is greater than it, the corresponding element in the destination vector is
	set to all ones. Otherwise, it is set to all zeros--*/
	uint8x8_t vcgt_s8 (int8x8_t __a, int8x8_t __b);
	uint16x4_t vcgt_s16 (int16x4_t __a, int16x4_t __b);
	uint32x2_t vcgt_s32 (int32x2_t __a, int32x2_t __b);
	uint32x2_t vcgt_f32 (float32x2_t __a, float32x2_t __b);
	uint8x8_t vcgt_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4_t vcgt_u16 (uint16x4_t __a, uint16x4_t __b);
	uint32x2_t vcgt_u32 (uint32x2_t __a, uint32x2_t __b);
	uint8x16_t vcgtq_s8 (int8x16_t __a, int8x16_t __b);
	uint16x8_t vcgtq_s16 (int16x8_t __a, int16x8_t __b);
	uint32x4_t vcgtq_s32 (int32x4_t __a, int32x4_t __b);
	uint32x4_t vcgtq_f32 (float32x4_t __a, float32x4_t __b);
	uint8x16_t vcgtq_u8 (uint8x16_t __a, uint8x16_t __b);
	uint16x8_t vcgtq_u16 (uint16x8_t __a, uint16x8_t __b);
	uint32x4_t vcgtq_u32 (uint32x4_t __a, uint32x4_t __b);
	/*--5、Vector compare less-than(正常指令): vclt -> ri = ai < bi ? 1...1:0...0;
	If it is less than it, the corresponding element in the destination vector is set
	to all ones.Otherwise, it is set to all zeros--*/
	uint8x8_t vclt_s8 (int8x8_t __a, int8x8_t __b);
	uint16x4_t vclt_s16 (int16x4_t __a, int16x4_t __b);
	uint32x2_t vclt_s32 (int32x2_t __a, int32x2_t __b);
	uint32x2_t vclt_f32 (float32x2_t __a, float32x2_t __b);
	uint8x8_t vclt_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4_t vclt_u16 (uint16x4_t __a, uint16x4_t __b);
	uint32x2_t vclt_u32 (uint32x2_t __a, uint32x2_t __b);
	uint8x16_t vcltq_s8 (int8x16_t __a, int8x16_t __b);
	uint16x8_t vcltq_s16 (int16x8_t __a, int16x8_t __b);
	uint32x4_t vcltq_s32 (int32x4_t __a, int32x4_t __b);
	uint32x4_t vcltq_f32 (float32x4_t __a, float32x4_t __b);
	uint8x16_t vcltq_u8 (uint8x16_t __a, uint8x16_t __b);
	uint16x8_t vcltq_u16 (uint16x8_t __a, uint16x8_t __b);
	uint32x4_t vcltq_u32 (uint32x4_t __a, uint32x4_t __b);
	/*--6、Vector compare absolute greater-than or equal(正常指令):
	vcage -> ri = |ai| >= |bi| ? 1...1:0...0;
	compares the absolute value of each element in a vector with the absolute value of the
	corresponding element of a second vector. If it is greater than or equal to it,
	the corresponding element in the destination vector is set to all ones.
	Otherwise, it is set to all zeros.--*/
	uint32x2_t vcage_f32 (float32x2_t __a, float32x2_t __b);
	uint32x4_t vcageq_f32 (float32x4_t __a, float32x4_t __b);
	/*--7、Vector compare absolute less-than or equal(正常指令):
	vcale -> ri = |ai| <= |bi| ? 1...1:0...0;
	compares the absolute value of each element in a vector with the absolute value of the
	corresponding element of a second vector. If it is less than or equal to it,
	the corresponding element in the destination vector is set to all ones.
	Otherwise, it is set to all zeros--*/
	uint32x2_t vcale_f32 (float32x2_t __a, float32x2_t __b);
	uint32x4_t vcaleq_f32 (float32x4_t __a, float32x4_t __b);
	/*--8、Vector compare absolute greater-than(正常指令):
	vcage -> ri = |ai| > |bi| ? 1...1:0...0;
	compares the absolute value of each element in a vector with the absolute value of the
	corresponding element of a second vector. If it is greater than it,
	the corresponding element in the destination vector is set to all ones.
	Otherwise, it is set to all zeros.--*/
	uint32x2_t vcagt_f32 (float32x2_t __a, float32x2_t __b);
	uint32x4_t vcagtq_f32 (float32x4_t __a, float32x4_t __b);
	/*--9、Vector compare absolute less-than(正常指令):
	vcalt -> ri = |ai| < |bi| ? 1...1:0...0;
	compares the absolute value of each element in a vector with the absolute value of the
	corresponding element of a second vector.If it is less than it, the corresponding 
	element in the destination vector is set to all ones. Otherwise,it is set to all zeros--*/
	uint32x2_t vcalt_f32 (float32x2_t __a, float32x2_t __b);
	uint32x4_t vcaltq_f32 (float32x4_t __a, float32x4_t __b);
	/**********************************************Vector test bits*************************/
	/*--正常指令,vtst -> ri = (ai & bi != 0) ? 1...1:0...0;
	bitwise logical ANDs each element in a vector with the corresponding element of a second
	vector.If the result is not zero, the corresponding element in the destination vector
	is set to all ones. Otherwise, it is set to all zeros--*/
	uint8x8_t vtst_s8 (int8x8_t __a, int8x8_t __b);
	uint16x4_t vtst_s16 (int16x4_t __a, int16x4_t __b);
	uint32x2_t vtst_s32 (int32x2_t __a, int32x2_t __b);
	uint8x8_t vtst_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4_t vtst_u16 (uint16x4_t __a, uint16x4_t __b);
	uint32x2_t vtst_u32 (uint32x2_t __a, uint32x2_t __b);
	uint8x8_t vtst_p8 (poly8x8_t __a, poly8x8_t __b);
	uint8x16_t vtstq_s8 (int8x16_t __a, int8x16_t __b);
	uint16x8_t vtstq_s16 (int16x8_t __a, int16x8_t __b);
	uint32x4_t vtstq_s32 (int32x4_t __a, int32x4_t __b);
	uint8x16_t vtstq_u8 (uint8x16_t __a, uint8x16_t __b);
	uint16x8_t vtstq_u16 (uint16x8_t __a, uint16x8_t __b);
	uint32x4_t vtstq_u32 (uint32x4_t __a, uint32x4_t __b);
	uint8x16_t vtstq_p8 (poly8x16_t __a, poly8x16_t __b);
	/**********************************************Absolute difference**********************/
	/*--1、Absolute difference between the arguments(正常指令): vabd -> ri = |ai - bi|;
	returns the absolute values of the results--*/
	int8x8_t vabd_s8 (int8x8_t __a, int8x8_t __b);
	int16x4_t vabd_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vabd_s32 (int32x2_t __a, int32x2_t __b);
	float32x2_t vabd_f32 (float32x2_t __a, float32x2_t __b);
	uint8x8_t vabd_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4_t vabd_u16 (uint16x4_t __a, uint16x4_t __b);
	uint32x2_t vabd_u32 (uint32x2_t __a, uint32x2_t __b);
	int8x16_t vabdq_s8 (int8x16_t __a, int8x16_t __b);
	int16x8_t vabdq_s16 (int16x8_t __a, int16x8_t __b);
	int32x4_t vabdq_s32 (int32x4_t __a, int32x4_t __b);
	float32x4_t vabdq_f32 (float32x4_t __a, float32x4_t __b);
	uint8x16_t vabdq_u8 (uint8x16_t __a, uint8x16_t __b);
	uint16x8_t vabdq_u16 (uint16x8_t __a, uint16x8_t __b);
	uint32x4_t vabdq_u32 (uint32x4_t __a, uint32x4_t __b);
	/*--2、Absolute difference - long(长指令): vabdl -> ri = |ai - bi|;
	The elements in the result vector are wider--*/
	int16x8_t vabdl_s8 (int8x8_t __a, int8x8_t __b);
	int32x4_t vabdl_s16 (int16x4_t __a, int16x4_t __b);
	int64x2_t vabdl_s32 (int32x2_t __a, int32x2_t __b);
	uint16x8_t vabdl_u8 (uint8x8_t __a, uint8x8_t __b);
	uint32x4_t vabdl_u16 (uint16x4_t __a, uint16x4_t __b);
	uint64x2_t vabdl_u32 (uint32x2_t __a, uint32x2_t __b);
	/*--3、Absolute difference and accumulate: vaba -> ri = ai + |bi - ci|;--*/
	int8x8_t vaba_s8 (int8x8_t __a, int8x8_t __b, int8x8_t __c);
	int16x4_t vaba_s16 (int16x4_t __a, int16x4_t __b, int16x4_t __c);
	int32x2_t vaba_s32 (int32x2_t __a, int32x2_t __b, int32x2_t __c);
	uint8x8_t vaba_u8 (uint8x8_t __a, uint8x8_t __b, uint8x8_t __c);
	uint16x4_t vaba_u16 (uint16x4_t __a, uint16x4_t __b, uint16x4_t __c);
	uint32x2_t vaba_u32 (uint32x2_t __a, uint32x2_t __b, uint32x2_t __c);
	int8x16_t vabaq_s8 (int8x16_t __a, int8x16_t __b, int8x16_t __c);
	int16x8_t vabaq_s16 (int16x8_t __a, int16x8_t __b, int16x8_t __c);
	int32x4_t vabaq_s32 (int32x4_t __a, int32x4_t __b, int32x4_t __c);
	uint8x16_t vabaq_u8 (uint8x16_t __a, uint8x16_t __b, uint8x16_t __c);
	uint16x8_t vabaq_u16 (uint16x8_t __a, uint16x8_t __b, uint16x8_t __c);
	uint32x4_t vabaq_u32 (uint32x4_t __a, uint32x4_t __b, uint32x4_t __c);
	/*--4、Absolute difference and accumulate - long: vabal -> ri = ai + |bi - ci|;
	The elements in the result are wider--*/
	int16x8_t vabal_s8 (int16x8_t __a, int8x8_t __b, int8x8_t __c);
	int32x4_t vabal_s16 (int32x4_t __a, int16x4_t __b, int16x4_t __c);
	int64x2_t vabal_s32 (int64x2_t __a, int32x2_t __b, int32x2_t __c);
	uint16x8_t vabal_u8 (uint16x8_t __a, uint8x8_t __b, uint8x8_t __c);
	uint32x4_t vabal_u16 (uint32x4_t __a, uint16x4_t __b, uint16x4_t __c);
	uint64x2_t vabal_u32 (uint64x2_t __a, uint32x2_t __b, uint32x2_t __c);
	/***********************************************Max*************************************/
	/*--正常指令, vmax -> ri = ai >= bi ? ai : bi; returns the larger of each pair--*/
	int8x8_t vmax_s8 (int8x8_t __a, int8x8_t __b);//_mm_max_epi8
	int16x4_t vmax_s16 (int16x4_t __a, int16x4_t __b);//_mm_max_epi16
	int32x2_t vmax_s32 (int32x2_t __a, int32x2_t __b);//_mm_max_epi32
	float32x2_t vmax_f32 (float32x2_t __a, float32x2_t __b);//_mm_max_ps
	uint8x8_t vmax_u8 (uint8x8_t __a, uint8x8_t __b);//_mm_max_epu8
	uint16x4_t vmax_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_max_epu16
	uint32x2_t vmax_u32 (uint32x2_t __a, uint32x2_t __b);//_mm_max_epu32
	int8x16_t vmaxq_s8 (int8x16_t __a, int8x16_t __b);//_mm_max_epi8
	int16x8_t vmaxq_s16 (int16x8_t __a, int16x8_t __b);//_mm_max_epi16
	int32x4_t vmaxq_s32 (int32x4_t __a, int32x4_t __b);//_mm_max_epi32
	float32x4_t vmaxq_f32 (float32x4_t __a, float32x4_t __b);//_mm_max_ps
	uint8x16_t vmaxq_u8 (uint8x16_t __a, uint8x16_t __b);//_mm_max_epu8
	uint16x8_t vmaxq_u16 (uint16x8_t __a, uint16x8_t __b);//_mm_max_epu16
	uint32x4_t vmaxq_u32 (uint32x4_t __a, uint32x4_t __b);//_mm_max_epu32
	/****************************************************Min********************************/
	/*--正常指令, vmin -> ri = ai >= bi ? bi : ai; returns the smaller of each pair--*/
	int8x8_t vmin_s8 (int8x8_t __a, int8x8_t __b);//_mm_min_epi8
	int16x4_t vmin_s16 (int16x4_t __a, int16x4_t __b);//_mm_min_epi16
	int32x2_t vmin_s32 (int32x2_t __a, int32x2_t __b);//_mm_min_epi32
	float32x2_t vmin_f32 (float32x2_t __a, float32x2_t __b);//_mm_min_ps
	uint8x8_t vmin_u8 (uint8x8_t __a, uint8x8_t __b);//_mm_min_epu8
	uint16x4_t vmin_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_min_epu16
	uint32x2_t vmin_u32 (uint32x2_t __a, uint32x2_t __b);//_mm_min_epu32
	int8x16_t vminq_s8 (int8x16_t __a, int8x16_t __b);//_mm_min_epi8
	int16x8_t vminq_s16 (int16x8_t __a, int16x8_t __b);//_mm_min_epi16
	int32x4_t vminq_s32 (int32x4_t __a, int32x4_t __b);//_mm_min_epi32
	float32x4_t vminq_f32 (float32x4_t __a, float32x4_t __b);//_mm_min_ps
	uint8x16_t vminq_u8 (uint8x16_t __a, uint8x16_t __b);//_mm_min_epu8
	uint16x8_t vminq_u16 (uint16x8_t __a, uint16x8_t __b);//_mm_min_epu16
	uint32x4_t vminq_u32 (uint32x4_t __a, uint32x4_t __b);//_mm_min_epu32
	/*******************************************Pairwise addition***************************/
	/*--1、Pairwise add(正常指令): 
	vpadd -> r0 = a0 + a1, ..., r3 = a6 + a7, r4 = b0 + b1, ..., r7 = b6 + b7
	adds adjacent pairs of elements of two vectors, 
	and places the results in the destination vector.--*/
	//r0 = a0 + a1, ...,r3 = a6 + a7, r4 = b0 + b1, ...,r7 = b6 + b7
	int8x8_t vpadd_s8 (int8x8_t __a, int8x8_t __b);
	int16x4_t vpadd_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vpadd_s32 (int32x2_t __a, int32x2_t __b);
	float32x2_t vpadd_f32 (float32x2_t __a, float32x2_t __b);
	uint8x8_t vpadd_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4_t vpadd_u16 (uint16x4_t __a, uint16x4_t __b);
	uint32x2_t vpadd_u32 (uint32x2_t __a, uint32x2_t __b);
	/*--2、Long pairwise add: vpaddl vpaddl -> r0 = a0 + a1, ..., r3 = a6 + a7;
	adds adjacent pairs of elements of a vector, sign extends or zero extends the results to 
	twice their original width, and places the final results in the destination vector--*/
	int16x4_t vpaddl_s8 (int8x8_t __a);
	int32x2_t vpaddl_s16 (int16x4_t __a);
	int64x1_t vpaddl_s32 (int32x2_t __a);
	uint16x4_t vpaddl_u8 (uint8x8_t __a);
	uint32x2_t vpaddl_u16 (uint16x4_t __a);
	uint64x1_t vpaddl_u32 (uint32x2_t __a);
	int16x8_t vpaddlq_s8 (int8x16_t __a);
	int32x4_t vpaddlq_s16 (int16x8_t __a);
	int64x2_t vpaddlq_s32 (int32x4_t __a);
	uint16x8_t vpaddlq_u8 (uint8x16_t __a);
	uint32x4_t vpaddlq_u16 (uint16x8_t __a);
	uint64x2_t vpaddlq_u32 (uint32x4_t __a);
	/*--3、Long pairwise add and accumulate: 
	vpadal -> r0 = a0 + (b0 + b1), ..., r3 = a3 + (b6 + b7);
	adds adjacent pairs of elements in the second vector, sign extends or zero extends the
	results to twice the original width.  It then accumulates this with the corresponding 
	element in the first vector and places the final results in the destination vector--*/
	int16x4_t vpadal_s8 (int16x4_t __a, int8x8_t __b);
	int32x2_t vpadal_s16 (int32x2_t __a, int16x4_t __b);
	int64x1_t vpadal_s32 (int64x1_t __a, int32x2_t __b);
	uint16x4_t vpadal_u8 (uint16x4_t __a, uint8x8_t __b);
	uint32x2_t vpadal_u16 (uint32x2_t __a, uint16x4_t __b);
	uint64x1_t vpadal_u32 (uint64x1_t __a, uint32x2_t __b);
	int16x8_t vpadalq_s8 (int16x8_t __a, int8x16_t __b);
	int32x4_t vpadalq_s16 (int32x4_t __a, int16x8_t __b);
	int64x2_t vpadalq_s32 (int64x2_t __a, int32x4_t __b);
	uint16x8_t vpadalq_u8 (uint16x8_t __a, uint8x16_t __b);
	uint32x4_t vpadalq_u16 (uint32x4_t __a, uint16x8_t __b);
	uint64x2_t vpadalq_u32 (uint64x2_t __a, uint32x4_t __b);
	/**********************************************Folding maximum**************************/
	/*--饱和指令, vpmax -> vpmax r0 = a0 >= a1 ? a0 : a1, ..., r4 = b0 >= b1 ? b0 : b1, ...;
	compares adjacent pairs of elements, and copies the larger of each pair into the 
	destination vector.The maximums from each pair of the first input vector are stored in 
	the lower half of the destination vector. The maximums from each pair of the second input 
	vector are stored in the higher half of the destination vector--*/
	int8x8_t vpmax_s8 (int8x8_t __a, int8x8_t __b);
	int16x4_t vpmax_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vpmax_s32 (int32x2_t __a, int32x2_t __b);
	float32x2_t vpmax_f32 (float32x2_t __a, float32x2_t __b);
	uint8x8_t vpmax_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4_t vpmax_u16 (uint16x4_t __a, uint16x4_t __b);
	uint32x2_t vpmax_u32 (uint32x2_t __a, uint32x2_t __b);
	/***************************************************Folding minimum*********************/
	/*--饱和指令, vpmin -> r0 = a0 >= a1 ? a1 : a0, ..., r4 = b0 >= b1 ? b1 : b0, ...;
	compares adjacent pairs of elements, and copies the smaller of each pair into the 
	destination vector.The minimums from each pair of the first input vector are stored in 
	the lower half of the destination vector. The minimums from each pair of the second 
	input vector are stored in the higher half of the destination vector.--*/
	int8x8_t vpmin_s8 (int8x8_t __a, int8x8_t __b);
	int16x4_t vpmin_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vpmin_s32 (int32x2_t __a, int32x2_t __b);
	float32x2_t vpmin_f32 (float32x2_t __a, float32x2_t __b);
	uint8x8_t vpmin_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4_t vpmin_u16 (uint16x4_t __a, uint16x4_t __b);
	uint32x2_t vpmin_u32 (uint32x2_t __a, uint32x2_t __b);
	/***************************************************Reciprocal**************************/
	/*--1、饱和指令, Newton-Raphson iteration(牛顿 - 拉夫逊迭代)
	performs a Newton-Raphson step for finding the reciprocal. It multiplies the elements of
	one vector by the corresponding elements of another vector, subtracts each of the results
	from 2, and places the final results into the elements of the destination vector--*/
	float32x2_t vrecps_f32 (float32x2_t __a, float32x2_t __b);
	float32x4_t vrecpsq_f32 (float32x4_t __a, float32x4_t __b);
	/*--2、饱和指令,performs a Newton-Raphson step for finding the reciprocal square root. 
	It multiplies the elements of one vector by the corresponding elements of another vector, 
	subtracts each of the results from 3, divides these results by two, and places 
	the final results into the elements of the destination vector--*/
	float32x2_t vrsqrts_f32 (float32x2_t __a, float32x2_t __b);
	float32x4_t vrsqrtsq_f32 (float32x4_t __a, float32x4_t __b);
	/************************************************Shifts by signed variable**************/
	/*--1、Vector shift left(饱和指令): vshl -> ri = ai << bi; (negative values shift right)
	left shifts each element in a vector by an amount specified in the corresponding element 
	in the second input vector. The shift amount is the signed integer value of the least 
	significant byte of the element in the second input vector. The bits shifted out of each
	element are lost.If the signed integer value is negative, it results in a right shift--*/
	int8x8_t vshl_s8 (int8x8_t __a, int8x8_t __b);
	int16x4_t vshl_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vshl_s32 (int32x2_t __a, int32x2_t __b);
	int64x1_t vshl_s64 (int64x1_t __a, int64x1_t __b);
	uint8x8_t vshl_u8 (uint8x8_t __a, int8x8_t __b);
	uint16x4_t vshl_u16 (uint16x4_t __a, int16x4_t __b);
	uint32x2_t vshl_u32 (uint32x2_t __a, int32x2_t __b);
	uint64x1_t vshl_u64 (uint64x1_t __a, int64x1_t __b);
	int8x16_t vshlq_s8 (int8x16_t __a, int8x16_t __b);
	int16x8_t vshlq_s16 (int16x8_t __a, int16x8_t __b);
	int32x4_t vshlq_s32 (int32x4_t __a, int32x4_t __b);
	int64x2_t vshlq_s64 (int64x2_t __a, int64x2_t __b);
	uint8x16_t vshlq_u8 (uint8x16_t __a, int8x16_t __b);
	uint16x8_t vshlq_u16 (uint16x8_t __a, int16x8_t __b);
	uint32x4_t vshlq_u32 (uint32x4_t __a, int32x4_t __b);
	uint64x2_t vshlq_u64 (uint64x2_t __a, int64x2_t __b);
	/*--2、Vector saturating shift left(饱和指令): 
	vqshl -> ri = ai << bi;(negative values shift right)
	If the shift value is positive, the operation is a left shift. Otherwise, it is a 
	truncating right shift. left shifts each element in a vector of integers and places
	the results in the destination vector. It is similar to VSHL. 
	The difference is that the sticky QC flag is set if saturation occurs--*/
	int8x8_t vqshl_s8 (int8x8_t __a, int8x8_t __b);
	int16x4_t vqshl_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vqshl_s32 (int32x2_t __a, int32x2_t __b);
	int64x1_t vqshl_s64 (int64x1_t __a, int64x1_t __b);
	uint8x8_t vqshl_u8 (uint8x8_t __a, int8x8_t __b);
	uint16x4_t vqshl_u16 (uint16x4_t __a, int16x4_t __b);
	uint32x2_t vqshl_u32 (uint32x2_t __a, int32x2_t __b);
	uint64x1_t vqshl_u64 (uint64x1_t __a, int64x1_t __b);
	int8x16_t vqshlq_s8 (int8x16_t __a, int8x16_t __b);
	int16x8_t vqshlq_s16 (int16x8_t __a, int16x8_t __b);
	int32x4_t vqshlq_s32 (int32x4_t __a, int32x4_t __b);
	int64x2_t vqshlq_s64 (int64x2_t __a, int64x2_t __b);
	uint8x16_t vqshlq_u8 (uint8x16_t __a, int8x16_t __b);
	uint16x8_t vqshlq_u16 (uint16x8_t __a, int16x8_t __b);
	uint32x4_t vqshlq_u32 (uint32x4_t __a, int32x4_t __b);
	uint64x2_t vqshlq_u64 (uint64x2_t __a, int64x2_t __b);
	/*--3、Vector rounding shift left(饱和指令): 
	vrshl -> ri = ai << bi;(negative values shift right)
	If the shift value is positive, the operation is a left shift. Otherwise, it is a
	rounding right shift. left shifts each element in a vector of integers and places
	the results in the destination vector. It is similar to VSHL. 
	The difference is that the shifted value is then rounded.--*/
	int8x8_t vrshl_s8 (int8x8_t __a, int8x8_t __b);
	int16x4_t vrshl_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vrshl_s32 (int32x2_t __a, int32x2_t __b);
	int64x1_t vrshl_s64 (int64x1_t __a, int64x1_t __b);
	uint8x8_t vrshl_u8 (uint8x8_t __a, int8x8_t __b);
	uint16x4_t vrshl_u16 (uint16x4_t __a, int16x4_t __b);
	uint32x2_t vrshl_u32 (uint32x2_t __a, int32x2_t __b);
	uint64x1_t vrshl_u64 (uint64x1_t __a, int64x1_t __b);
	int8x16_t vrshlq_s8 (int8x16_t __a, int8x16_t __b);
	int16x8_t vrshlq_s16 (int16x8_t __a, int16x8_t __b);
	int32x4_t vrshlq_s32 (int32x4_t __a, int32x4_t __b);
	int64x2_t vrshlq_s64 (int64x2_t __a, int64x2_t __b);
	uint8x16_t vrshlq_u8 (uint8x16_t __a, int8x16_t __b);
	uint16x8_t vrshlq_u16 (uint16x8_t __a, int16x8_t __b);
	uint32x4_t vrshlq_u32 (uint32x4_t __a, int32x4_t __b);
	uint64x2_t vrshlq_u64 (uint64x2_t __a, int64x2_t __b);
	/*--4、Vector saturating rounding shift left(饱和指令):
	vqrshl -> ri = ai << bi;(negative values shift right)
	left shifts each element in a vector of integers and places the results in the 
	destination vector.It is similar to VSHL. The difference is that the shifted value
	is rounded, and the sticky QC flag is set if saturation occurs.--*/
	int8x8_t vqrshl_s8 (int8x8_t __a, int8x8_t __b);
	int16x4_t vqrshl_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vqrshl_s32 (int32x2_t __a, int32x2_t __b);
	int64x1_t vqrshl_s64 (int64x1_t __a, int64x1_t __b);
	uint8x8_t vqrshl_u8 (uint8x8_t __a, int8x8_t __b);
	uint16x4_t vqrshl_u16 (uint16x4_t __a, int16x4_t __b);
	uint32x2_t vqrshl_u32 (uint32x2_t __a, int32x2_t __b);
	uint64x1_t vqrshl_u64 (uint64x1_t __a, int64x1_t __b);
	int8x16_t vqrshlq_s8 (int8x16_t __a, int8x16_t __b);
	int16x8_t vqrshlq_s16 (int16x8_t __a, int16x8_t __b);
	int32x4_t vqrshlq_s32 (int32x4_t __a, int32x4_t __b);
	int64x2_t vqrshlq_s64 (int64x2_t __a, int64x2_t __b);
	uint8x16_t vqrshlq_u8 (uint8x16_t __a, int8x16_t __b);
	uint16x8_t vqrshlq_u16 (uint16x8_t __a, int16x8_t __b);
	uint32x4_t vqrshlq_u32 (uint32x4_t __a, int32x4_t __b);
	uint64x2_t vqrshlq_u64 (uint64x2_t __a, int64x2_t __b);
	/****************************************Shifts by a constant***************************/
	/*--1、Vector shift right by constant: vshr -> ri = ai >> b;The results are truncated.
	right shifts each element in a vector by an immediate value, 
	and places the results in the destination vector.--*/
	int8x8_t vshr_n_s8 (int8x8_t __a, const int __b);
	int16x4_t vshr_n_s16 (int16x4_t __a, const int __b);
	int32x2_t vshr_n_s32 (int32x2_t __a, const int __b);
	int64x1_t vshr_n_s64 (int64x1_t __a, const int __b);
	uint8x8_t vshr_n_u8 (uint8x8_t __a, const int __b);
	uint16x4_t vshr_n_u16 (uint16x4_t __a, const int __b);
	uint32x2_t vshr_n_u32 (uint32x2_t __a, const int __b);
	uint64x1_t vshr_n_u64 (uint64x1_t __a, const int __b);
	int8x16_t vshrq_n_s8 (int8x16_t __a, const int __b);
	int16x8_t vshrq_n_s16 (int16x8_t __a, const int __b);
	int32x4_t vshrq_n_s32 (int32x4_t __a, const int __b);
	int64x2_t vshrq_n_s64 (int64x2_t __a, const int __b);
	uint8x16_t vshrq_n_u8 (uint8x16_t __a, const int __b);
	uint16x8_t vshrq_n_u16 (uint16x8_t __a, const int __b);
	uint32x4_t vshrq_n_u32 (uint32x4_t __a, const int __b);
	uint64x2_t vshrq_n_u64 (uint64x2_t __a, const int __b);
	/*--2、Vector shift left by constant: vshl -> ri = ai << b;
	left shifts each element in a vector by an immediate value, and places the results in the 
	destination vector. The bits shifted out of the left of each element are lost--*/
	int8x8_t vshl_n_s8 (int8x8_t __a, const int __b);
	int16x4_t vshl_n_s16 (int16x4_t __a, const int __b);
	int32x2_t vshl_n_s32 (int32x2_t __a, const int __b);
	int64x1_t vshl_n_s64 (int64x1_t __a, const int __b);
	uint8x8_t vshl_n_u8 (uint8x8_t __a, const int __b);
	uint16x4_t vshl_n_u16 (uint16x4_t __a, const int __b);
	uint32x2_t vshl_n_u32 (uint32x2_t __a, const int __b);
	uint64x1_t vshl_n_u64 (uint64x1_t __a, const int __b);
	int8x16_t vshlq_n_s8 (int8x16_t __a, const int __b);
	int16x8_t vshlq_n_s16 (int16x8_t __a, const int __b);
	int32x4_t vshlq_n_s32 (int32x4_t __a, const int __b);
	int64x2_t vshlq_n_s64 (int64x2_t __a, const int __b);
	uint8x16_t vshlq_n_u8 (uint8x16_t __a, const int __b);
	uint16x8_t vshlq_n_u16 (uint16x8_t __a, const int __b);
	uint32x4_t vshlq_n_u32 (uint32x4_t __a, const int __b);
	uint64x2_t vshlq_n_u64 (uint64x2_t __a, const int __b);
	/*--3、Vector rounding shift right by constant: vrshr -> ri = ai >> b;
	right shifts each element in a vector by an immediate value, and places the results
	in the destination vector. The shifted values are rounded.--*/
	int8x8_t vrshr_n_s8 (int8x8_t __a, const int __b);
	int16x4_t vrshr_n_s16 (int16x4_t __a, const int __b);
	int32x2_t vrshr_n_s32 (int32x2_t __a, const int __b);
	int64x1_t vrshr_n_s64 (int64x1_t __a, const int __b);
	uint8x8_t vrshr_n_u8 (uint8x8_t __a, const int __b);
	uint16x4_t vrshr_n_u16 (uint16x4_t __a, const int __b);
	uint32x2_t vrshr_n_u32 (uint32x2_t __a, const int __b);
	uint64x1_t vrshr_n_u64 (uint64x1_t __a, const int __b);
	int8x16_t vrshrq_n_s8 (int8x16_t __a, const int __b);
	int16x8_t vrshrq_n_s16 (int16x8_t __a, const int __b);
	int32x4_t vrshrq_n_s32 (int32x4_t __a, const int __b);
	int64x2_t vrshrq_n_s64 (int64x2_t __a, const int __b);
	uint8x16_t vrshrq_n_u8 (uint8x16_t __a, const int __b);
	uint16x8_t vrshrq_n_u16 (uint16x8_t __a, const int __b);
	uint32x4_t vrshrq_n_u32 (uint32x4_t __a, const int __b);
	uint64x2_t vrshrq_n_u64 (uint64x2_t __a, const int __b);
	/*--4、Vector shift right by constant and accumulate: vsra -> ri = (ai >> c) + (bi >> c); 
	The results are truncated. right shifts each element in a vector by an immediate value, 
	and accumulates the results into the destination vector.--*/
	int8x8_t vsra_n_s8 (int8x8_t __a, int8x8_t __b, const int __c);
	int16x4_t vsra_n_s16 (int16x4_t __a, int16x4_t __b, const int __c);
	int32x2_t vsra_n_s32 (int32x2_t __a, int32x2_t __b, const int __c);
	int64x1_t vsra_n_s64 (int64x1_t __a, int64x1_t __b, const int __c);
	uint8x8_t vsra_n_u8 (uint8x8_t __a, uint8x8_t __b, const int __c);
	uint16x4_t vsra_n_u16 (uint16x4_t __a, uint16x4_t __b, const int __c);
	uint32x2_t vsra_n_u32 (uint32x2_t __a, uint32x2_t __b, const int __c);
	uint64x1_t vsra_n_u64 (uint64x1_t __a, uint64x1_t __b, const int __c);
	int8x16_t vsraq_n_s8 (int8x16_t __a, int8x16_t __b, const int __c);
	int16x8_t vsraq_n_s16 (int16x8_t __a, int16x8_t __b, const int __c);
	int32x4_t vsraq_n_s32 (int32x4_t __a, int32x4_t __b, const int __c);
	int64x2_t vsraq_n_s64 (int64x2_t __a, int64x2_t __b, const int __c);
	uint8x16_t vsraq_n_u8 (uint8x16_t __a, uint8x16_t __b, const int __c);
	uint16x8_t vsraq_n_u16 (uint16x8_t __a, uint16x8_t __b, const int __c);
	uint32x4_t vsraq_n_u32 (uint32x4_t __a, uint32x4_t __b, const int __c);
	uint64x2_t vsraq_n_u64 (uint64x2_t __a, uint64x2_t __b, const int __c);
	/*--5、Vector rounding shift right by constant and accumulate: 
	vrsra -> ri = (ai >> c) + (bi >> c);
	The results are rounded.right shifts each element in a vector by an immediate value, 
	and accumulates the rounded results into the destination vector.--*/
	int8x8_t vrsra_n_s8 (int8x8_t __a, int8x8_t __b, const int __c);
	int16x4_t vrsra_n_s16 (int16x4_t __a, int16x4_t __b, const int __c);
	int32x2_t vrsra_n_s32 (int32x2_t __a, int32x2_t __b, const int __c);
	int64x1_t vrsra_n_s64 (int64x1_t __a, int64x1_t __b, const int __c);
	uint8x8_t vrsra_n_u8 (uint8x8_t __a, uint8x8_t __b, const int __c);
	uint16x4_t vrsra_n_u16 (uint16x4_t __a, uint16x4_t __b, const int __c);
	uint32x2_t vrsra_n_u32 (uint32x2_t __a, uint32x2_t __b, const int __c);
	uint64x1_t vrsra_n_u64 (uint64x1_t __a, uint64x1_t __b, const int __c);
	int8x16_t vrsraq_n_s8 (int8x16_t __a, int8x16_t __b, const int __c);
	int16x8_t vrsraq_n_s16 (int16x8_t __a, int16x8_t __b, const int __c);
	int32x4_t vrsraq_n_s32 (int32x4_t __a, int32x4_t __b, const int __c);
	int64x2_t vrsraq_n_s64 (int64x2_t __a, int64x2_t __b, const int __c);
	uint8x16_t vrsraq_n_u8 (uint8x16_t __a, uint8x16_t __b, const int __c);
	uint16x8_t vrsraq_n_u16 (uint16x8_t __a, uint16x8_t __b, const int __c);
	uint32x4_t vrsraq_n_u32 (uint32x4_t __a, uint32x4_t __b, const int __c);
	uint64x2_t vrsraq_n_u64 (uint64x2_t __a, uint64x2_t __b, const int __c);
	/*--6、Vector saturating shift left by constant: vqshl -> ri = sat(ai << b); 
	left shifts each element in a vector of integers by an immediate value, and places the 
	results in the destination vector,and the sticky QC flag is set if saturation occurs.--*/
	int8x8_t vqshl_n_s8 (int8x8_t __a, const int __b);
	int16x4_t vqshl_n_s16 (int16x4_t __a, const int __b);
	int32x2_t vqshl_n_s32 (int32x2_t __a, const int __b);
	int64x1_t vqshl_n_s64 (int64x1_t __a, const int __b);
	uint8x8_t vqshl_n_u8 (uint8x8_t __a, const int __b);
	uint16x4_t vqshl_n_u16 (uint16x4_t __a, const int __b);
	uint32x2_t vqshl_n_u32 (uint32x2_t __a, const int __b);
	uint64x1_t vqshl_n_u64 (uint64x1_t __a, const int __b);
	int8x16_t vqshlq_n_s8 (int8x16_t __a, const int __b);
	int16x8_t vqshlq_n_s16 (int16x8_t __a, const int __b);
	int32x4_t vqshlq_n_s32 (int32x4_t __a, const int __b);
	int64x2_t vqshlq_n_s64 (int64x2_t __a, const int __b);
	uint8x16_t vqshlq_n_u8 (uint8x16_t __a, const int __b);
	uint16x8_t vqshlq_n_u16 (uint16x8_t __a, const int __b);
	uint32x4_t vqshlq_n_u32 (uint32x4_t __a, const int __b);
	uint64x2_t vqshlq_n_u64 (uint64x2_t __a, const int __b);
	/*--7、Vector signed->unsigned saturating shift left by constant: vqshlu -> ri = ai << b; 
	left shifts each element in a vector of integers by an immediate value, places the 
	results in the destination vector, the sticky QC flag is set if saturation occurs, 
	and indicates that the results are unsigned even though the operands are signed.--*/
	uint8x8_t vqshlu_n_s8 (int8x8_t __a, const int __b);
	uint16x4_t vqshlu_n_s16 (int16x4_t __a, const int __b);
	uint32x2_t vqshlu_n_s32 (int32x2_t __a, const int __b);
	uint64x1_t vqshlu_n_s64 (int64x1_t __a, const int __b);
	uint8x16_t vqshluq_n_s8 (int8x16_t __a, const int __b);
	uint16x8_t vqshluq_n_s16 (int16x8_t __a, const int __b);
	uint32x4_t vqshluq_n_s32 (int32x4_t __a, const int __b);
	uint64x2_t vqshluq_n_s64 (int64x2_t __a, const int __b);
	/*--8、Vector narrowing shift right by constant: vshrn -> ri = ai >> b;
	The results are truncated.right shifts each element in the input vector by an 
	immediate value. It then narrows the result by storing only the least significant
	half of each element into the destination vector.--*/
	int8x8_t vshrn_n_s16 (int16x8_t __a, const int __b);
	int16x4_t vshrn_n_s32 (int32x4_t __a, const int __b);
	int32x2_t vshrn_n_s64 (int64x2_t __a, const int __b);
	uint8x8_t vshrn_n_u16 (uint16x8_t __a, const int __b);
	uint16x4_t vshrn_n_u32 (uint32x4_t __a, const int __b);
	uint32x2_t vshrn_n_u64 (uint64x2_t __a, const int __b);
	/*--9、Vector signed->unsigned narrowing saturating shift right by constant: 
	vqshrun -> ri = ai >> b; 
	Results are truncated. right shifts each element in a quadword vector of integers by an
	immediate value, and places the results in a doubleword vector. The results are unsigned, 
	although the operands are signed. The sticky QC flag is set if saturation occurs.--*/
	uint8x8_t vqshrun_n_s16 (int16x8_t __a, const int __b);
	uint16x4_t vqshrun_n_s32 (int32x4_t __a, const int __b);
	uint32x2_t vqshrun_n_s64 (int64x2_t __a, const int __b);
	/*--10、Vector signed->unsigned rounding narrowing saturating shift right by constant: 
	vqrshrun -> ri = ai >> b; Results are rounded. right shifts each element in a quadword 
	vector of integers by an immediate value, and places the rounded results in a doubleword 
	vector. The results are unsigned, although the operands are signed.--*/
	uint8x8_t vqrshrun_n_s16 (int16x8_t __a, const int __b);
	uint16x4_t vqrshrun_n_s32 (int32x4_t __a, const int __b);
	uint32x2_t vqrshrun_n_s64 (int64x2_t __a, const int __b);
	/*--11、Vector narrowing saturating shift right by constant: vqshrn -> ri = ai >> b; 
	Results are truncated. right shifts each element in a quadword vector of integers by an 
	immediate value, and places the results in a doubleword vector, 
	and the sticky QC flag is set if saturation occurs.--*/
	int8x8_t vqshrn_n_s16 (int16x8_t __a, const int __b);
	int16x4_t vqshrn_n_s32 (int32x4_t __a, const int __b);
	int32x2_t vqshrn_n_s64 (int64x2_t __a, const int __b);
	uint8x8_t vqshrn_n_u16 (uint16x8_t __a, const int __b);
	uint16x4_t vqshrn_n_u32 (uint32x4_t __a, const int __b);
	uint32x2_t vqshrn_n_u64 (uint64x2_t __a, const int __b);
	/*--12、Vector rounding narrowing shift right by constant: vrshrn -> ri = ai >> b; 
	The results are rounded. right shifts each element in a vector by an immediate value,
	and places the rounded,narrowed results in the destination vector.--*/
	int8x8_t vrshrn_n_s16 (int16x8_t __a, const int __b);
	int16x4_t vrshrn_n_s32 (int32x4_t __a, const int __b);
	int32x2_t vrshrn_n_s64 (int64x2_t __a, const int __b);
	uint8x8_t vrshrn_n_u16 (uint16x8_t __a, const int __b);
	uint16x4_t vrshrn_n_u32 (uint32x4_t __a, const int __b);
	uint32x2_t vrshrn_n_u64 (uint64x2_t __a, const int __b);
	/*--13、Vector rounding narrowing saturating shift right by constant:
	vqrshrn -> ri = ai >> b;
	Results are rounded. right shifts each element in a quadword vector of integers by an 
	immediate value,and places the rounded,narrowed results in a doubleword vector. 
	The sticky QC flag is set if saturation occurs.--*/
	int8x8_t vqrshrn_n_s16 (int16x8_t __a, const int __b);
	int16x4_t vqrshrn_n_s32 (int32x4_t __a, const int __b);
	int32x2_t vqrshrn_n_s64 (int64x2_t __a, const int __b);
	uint8x8_t vqrshrn_n_u16 (uint16x8_t __a, const int __b);
	uint16x4_t vqrshrn_n_u32 (uint32x4_t __a, const int __b);
	uint32x2_t vqrshrn_n_u64 (uint64x2_t __a, const int __b);
	/*--14、Vector widening shift left by constant: vshll -> ri = ai << b; 
	left shifts each element in a vector of integers by an immediate value, 
	and place the results in the destination vector. Bits shifted out of the left of each
	element are lost and values are sign extended or zero extended.--*/
	int16x8_t vshll_n_s8 (int8x8_t __a, const int __b);
	int32x4_t vshll_n_s16 (int16x4_t __a, const int __b);
	int64x2_t vshll_n_s32 (int32x2_t __a, const int __b);
	uint16x8_t vshll_n_u8 (uint8x8_t __a, const int __b);
	uint32x4_t vshll_n_u16 (uint16x4_t __a, const int __b);
	uint64x2_t vshll_n_u32 (uint32x2_t __a, const int __b);
	/********************************************Shifts with insert*************************/
	/*--1、Vector shift right and insert: vsri -> ; The two most significant bits in the 
	destination vector are unchanged. right shifts each element in the second input vector 
	by an immediate value, and inserts the results in the destination vector. It does not 
	affect the highest n significant bits of the elements in the destination register.
	Bits shifted out of the right of each element are lost.The first input vector holds
	the elements of the destination vector before the operation is performed.--*/
	int8x8_t vsri_n_s8 (int8x8_t __a, int8x8_t __b, const int __c);
	int16x4_t vsri_n_s16 (int16x4_t __a, int16x4_t __b, const int __c);
	int32x2_t vsri_n_s32 (int32x2_t __a, int32x2_t __b, const int __c);
	int64x1_t vsri_n_s64 (int64x1_t __a, int64x1_t __b, const int __c);
	uint8x8_t vsri_n_u8 (uint8x8_t __a, uint8x8_t __b, const int __c);
	uint16x4_t vsri_n_u16 (uint16x4_t __a, uint16x4_t __b, const int __c);
	uint32x2_t vsri_n_u32 (uint32x2_t __a, uint32x2_t __b, const int __c);
	uint64x1_t vsri_n_u64 (uint64x1_t __a, uint64x1_t __b, const int __c);
	poly8x8_t vsri_n_p8 (poly8x8_t __a, poly8x8_t __b, const int __c);
	poly16x4_t vsri_n_p16 (poly16x4_t __a, poly16x4_t __b, const int __c);
	int8x16_t vsriq_n_s8 (int8x16_t __a, int8x16_t __b, const int __c);
	int16x8_t vsriq_n_s16 (int16x8_t __a, int16x8_t __b, const int __c);
	int32x4_t vsriq_n_s32 (int32x4_t __a, int32x4_t __b, const int __c);
	int64x2_t vsriq_n_s64 (int64x2_t __a, int64x2_t __b, const int __c);
	uint8x16_t vsriq_n_u8 (uint8x16_t __a, uint8x16_t __b, const int __c);
	uint16x8_t vsriq_n_u16 (uint16x8_t __a, uint16x8_t __b, const int __c);
	uint32x4_t vsriq_n_u32 (uint32x4_t __a, uint32x4_t __b, const int __c);
	uint64x2_t vsriq_n_u64 (uint64x2_t __a, uint64x2_t __b, const int __c);
	poly8x16_t vsriq_n_p8 (poly8x16_t __a, poly8x16_t __b, const int __c);
	poly16x8_t vsriq_n_p16 (poly16x8_t __a, poly16x8_t __b, const int __c);
	/*--2、Vector shift left and insert: vsli ->; The least significant bit in each element
	in the destination vector is unchanged. left shifts each element in the second input 
	vector by an immediate value, and inserts the results in the destination vector.
	It does not affect the lowest n significant bits of the elements in the destination 
	register. Bits shifted out of the left of each element are lost. The first input vector
	holds the elements of the destination vector before the operation is performed.--*/
	int8x8_t vsli_n_s8 (int8x8_t __a, int8x8_t __b, const int __c);
	int16x4_t vsli_n_s16 (int16x4_t __a, int16x4_t __b, const int __c);
	int32x2_t vsli_n_s32 (int32x2_t __a, int32x2_t __b, const int __c);
	int64x1_t vsli_n_s64 (int64x1_t __a, int64x1_t __b, const int __c);
	uint8x8_t vsli_n_u8 (uint8x8_t __a, uint8x8_t __b, const int __c);
	uint16x4_t vsli_n_u16 (uint16x4_t __a, uint16x4_t __b, const int __c);
	uint32x2_t vsli_n_u32 (uint32x2_t __a, uint32x2_t __b, const int __c);
	uint64x1_t vsli_n_u64 (uint64x1_t __a, uint64x1_t __b, const int __c);
	poly8x8_t vsli_n_p8 (poly8x8_t __a, poly8x8_t __b, const int __c);
	poly16x4_t vsli_n_p16 (poly16x4_t __a, poly16x4_t __b, const int __c);
	int8x16_t vsliq_n_s8 (int8x16_t __a, int8x16_t __b, const int __c);
	int16x8_t vsliq_n_s16 (int16x8_t __a, int16x8_t __b, const int __c);
	int32x4_t vsliq_n_s32 (int32x4_t __a, int32x4_t __b, const int __c);
	int64x2_t vsliq_n_s64 (int64x2_t __a, int64x2_t __b, const int __c);
	uint8x16_t vsliq_n_u8 (uint8x16_t __a, uint8x16_t __b, const int __c);
	uint16x8_t vsliq_n_u16 (uint16x8_t __a, uint16x8_t __b, const int __c);
	uint32x4_t vsliq_n_u32 (uint32x4_t __a, uint32x4_t __b, const int __c);
	uint64x2_t vsliq_n_u64 (uint64x2_t __a, uint64x2_t __b, const int __c);
	poly8x16_t vsliq_n_p8 (poly8x16_t __a, poly8x16_t __b, const int __c);
	poly16x8_t vsliq_n_p16 (poly16x8_t __a, poly16x8_t __b, const int __c);
	/*****************************************Absolute value********************************/
	/*--1、Absolute(正常指令): vabs -> ri = |ai|;
	returns the absolute value of each element in a vector.--*/
	int8x8_t vabs_s8 (int8x8_t __a);//_mm_abs_epi8
	int16x4_t vabs_s16 (int16x4_t __a);//_mm_abs_epi16
	int32x2_t vabs_s32 (int32x2_t __a);//_mm_abs_epi32
	float32x2_t vabs_f32 (float32x2_t __a);
	int8x16_t vabsq_s8 (int8x16_t __a);//_mm_abs_epi8
	int16x8_t vabsq_s16 (int16x8_t __a);//_mm_abs_epi16
	int32x4_t vabsq_s32 (int32x4_t __a);//_mm_abs_epi32
	float32x4_t vabsq_f32 (float32x4_t __a);
	/*--2、Saturating absolute(饱和指令): vqabs -> ri = sat(|ai|);
	returns the absolute value of each element in a vector. If any of the results overflow,
	they are saturated and the sticky QC flag is set.--*/
	int8x8_t vqabs_s8 (int8x8_t __a);
	int16x4_t vqabs_s16 (int16x4_t __a);
	int32x2_t vqabs_s32 (int32x2_t __a);
	int8x16_t vqabsq_s8 (int8x16_t __a);
	int16x8_t vqabsq_s16 (int16x8_t __a);
	int32x4_t vqabsq_s32 (int32x4_t __a);
	/***************************************************Negation****************************/
	/*--1、Negate(正常指令): vneg -> ri = -ai; negates each element in a vector.--*/
	int8x8_t vneg_s8 (int8x8_t __a);
	int16x4_t vneg_s16 (int16x4_t __a);
	int32x2_t vneg_s32 (int32x2_t __a);
	float32x2_t vneg_f32 (float32x2_t __a);
	int8x16_t vnegq_s8 (int8x16_t __a);
	int16x8_t vnegq_s16 (int16x8_t __a);
	int32x4_t vnegq_s32 (int32x4_t __a);
	float32x4_t vnegq_f32 (float32x4_t __a);
	/*--2、Saturating Negate: vqneg -> ri = sat(-ai);
	negates each element in a vector. If any of the results overflow, 
	they are saturated and the sticky QC flag is set.--*/
	int8x8_t vqneg_s8 (int8x8_t __a);
	int16x4_t vqneg_s16 (int16x4_t __a);
	int32x2_t vqneg_s32 (int32x2_t __a);
	int8x16_t vqnegq_s8 (int8x16_t __a);
	int16x8_t vqnegq_s16 (int16x8_t __a);
	int32x4_t vqnegq_s32 (int32x4_t __a);
	/********************************************Logical operations*************************/
	/*--1、Bitwise not(正常指令): vmvn -> ri = ~ai; 
	performs a bitwise inversion of each element from the input vector.--*/
	int8x8_t vmvn_s8 (int8x8_t __a);
	int16x4_t vmvn_s16 (int16x4_t __a);
	int32x2_t vmvn_s32 (int32x2_t __a);
	uint8x8_t vmvn_u8 (uint8x8_t __a);
	uint16x4_t vmvn_u16 (uint16x4_t __a);
	uint32x2_t vmvn_u32 (uint32x2_t __a);
	poly8x8_t vmvn_p8 (poly8x8_t __a);
	int8x16_t vmvnq_s8 (int8x16_t __a);
	int16x8_t vmvnq_s16 (int16x8_t __a);
	int32x4_t vmvnq_s32 (int32x4_t __a);
	uint8x16_t vmvnq_u8 (uint8x16_t __a);
	uint16x8_t vmvnq_u16 (uint16x8_t __a);
	uint32x4_t vmvnq_u32 (uint32x4_t __a);
	poly8x16_t vmvnq_p8 (poly8x16_t __a);
	/*--2、Bitwise and(正常指令): vand -> ri = ai & bi; performs a bitwise AND between 
	corresponding elements of the input vectors.--*/
	int8x8_t vand_s8 (int8x8_t __a, int8x8_t __b);//_mm_and_si128
	int16x4_t vand_s16 (int16x4_t __a, int16x4_t __b);//_mm_and_si128
	int32x2_t vand_s32 (int32x2_t __a, int32x2_t __b);//_mm_and_si128
	uint8x8_t vand_u8 (uint8x8_t __a, uint8x8_t __b);//_mm_and_si128
	uint16x4_t vand_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_and_si128
	uint32x2_t vand_u32 (uint32x2_t __a, uint32x2_t __b);//_mm_and_si128
	int64x1_t vand_s64 (int64x1_t __a, int64x1_t __b);//_mm_and_si128
	uint64x1_t vand_u64 (uint64x1_t __a, uint64x1_t __b);//_mm_and_si128
	int8x16_t vandq_s8 (int8x16_t __a, int8x16_t __b);//_mm_and_si128
	int16x8_t vandq_s16 (int16x8_t __a, int16x8_t __b);//_mm_and_si128
	int32x4_t vandq_s32 (int32x4_t __a, int32x4_t __b);//_mm_and_si128
	int64x2_t vandq_s64 (int64x2_t __a, int64x2_t __b);//_mm_and_si128
	uint8x16_t vandq_u8 (uint8x16_t __a, uint8x16_t __b);//_mm_and_si128
	uint16x8_t vandq_u16 (uint16x8_t __a, uint16x8_t __b);//_mm_and_si128
	uint32x4_t vandq_u32 (uint32x4_t __a, uint32x4_t __b);//_mm_and_si128
	uint64x2_t vandq_u64 (uint64x2_t __a, uint64x2_t __b);//_mm_and_si128
	/*--3、Bitwise or(正常指令): vorr -> ri = ai | bi; performs a bitwise OR between
	corresponding elements of the input vectors.--*/
	int8x8_t vorr_s8 (int8x8_t __a, int8x8_t __b);//_mm_or_si128
	int16x4_t vorr_s16 (int16x4_t __a, int16x4_t __b);//_mm_or_si128
	int32x2_t vorr_s32 (int32x2_t __a, int32x2_t __b);//_mm_or_si128
	uint8x8_t vorr_u8 (uint8x8_t __a, uint8x8_t __b);//_mm_or_si128
	uint16x4_t vorr_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_or_si128
	uint32x2_t vorr_u32 (uint32x2_t __a, uint32x2_t __b);//_mm_or_si128
	int64x1_t vorr_s64 (int64x1_t __a, int64x1_t __b);//_mm_or_si128
	uint64x1_t vorr_u64 (uint64x1_t __a, uint64x1_t __b);//_mm_or_si128
	int8x16_t vorrq_s8 (int8x16_t __a, int8x16_t __b);//_mm_or_si128
	int16x8_t vorrq_s16 (int16x8_t __a, int16x8_t __b);//_mm_or_si128
	int32x4_t vorrq_s32 (int32x4_t __a, int32x4_t __b);//_mm_or_si128
	int64x2_t vorrq_s64 (int64x2_t __a, int64x2_t __b);//_mm_or_si128
	uint8x16_t vorrq_u8 (uint8x16_t __a, uint8x16_t __b);//_mm_or_si128
	uint16x8_t vorrq_u16 (uint16x8_t __a, uint16x8_t __b);//_mm_or_si128
	uint32x4_t vorrq_u32 (uint32x4_t __a, uint32x4_t __b);//_mm_or_si128
	uint64x2_t vorrq_u64 (uint64x2_t __a, uint64x2_t __b);//_mm_or_si128
	/*--4、Bitwise exclusive or (EOR or XOR)(正常指令): veor -> ri = ai ^ bi; 
	performs a bitwise exclusive-OR between corresponding elements of the input vectors.--*/
	int8x8_t veor_s8 (int8x8_t __a, int8x8_t __b);//_mm_xor_si128
	int16x4_t veor_s16 (int16x4_t __a, int16x4_t __b);//_mm_xor_si128
	int32x2_t veor_s32 (int32x2_t __a, int32x2_t __b);//_mm_xor_si128
	uint8x8_t veor_u8 (uint8x8_t __a, uint8x8_t __b);//_mm_xor_si128
	uint16x4_t veor_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_xor_si128
	uint32x2_t veor_u32 (uint32x2_t __a, uint32x2_t __b);//_mm_xor_si128
	int64x1_t veor_s64 (int64x1_t __a, int64x1_t __b);//_mm_xor_si128
	uint64x1_t veor_u64 (uint64x1_t __a, uint64x1_t __b);//_mm_xor_si128
	int8x16_t veorq_s8 (int8x16_t __a, int8x16_t __b);//_mm_xor_si128
	int16x8_t veorq_s16 (int16x8_t __a, int16x8_t __b);//_mm_xor_si128
	int32x4_t veorq_s32 (int32x4_t __a, int32x4_t __b);//_mm_xor_si128
	int64x2_t veorq_s64 (int64x2_t __a, int64x2_t __b);//_mm_xor_si128
	uint8x16_t veorq_u8 (uint8x16_t __a, uint8x16_t __b);//_mm_xor_si128
	uint16x8_t veorq_u16 (uint16x8_t __a, uint16x8_t __b);//_mm_xor_si128
	uint32x4_t veorq_u32 (uint32x4_t __a, uint32x4_t __b);//_mm_xor_si128
	uint64x2_t veorq_u64 (uint64x2_t __a, uint64x2_t __b);//_mm_xor_si128
	/*--5、Bit Clear(正常指令): vbic -> ri = ~ai & bi;
	VBIC (Vector Bitwise Clear) performs a bitwise logical AND complement operation between
	values in two registers, and places the results in the destination register.--*/
	int8x8_t vbic_s8 (int8x8_t __a, int8x8_t __b);//_mm_andnot_si128
	int16x4_t vbic_s16 (int16x4_t __a, int16x4_t __b);//_mm_andnot_si128
	int32x2_t vbic_s32 (int32x2_t __a, int32x2_t __b);//_mm_andnot_si128
	uint8x8_t vbic_u8 (uint8x8_t __a, uint8x8_t __b);//_mm_andnot_si128
	uint16x4_t vbic_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_andnot_si128
	uint32x2_t vbic_u32 (uint32x2_t __a, uint32x2_t __b);//_mm_andnot_si128
	int64x1_t vbic_s64 (int64x1_t __a, int64x1_t __b);//_mm_andnot_si128
	uint64x1_t vbic_u64 (uint64x1_t __a, uint64x1_t __b);//_mm_andnot_si128
	int8x16_t vbicq_s8 (int8x16_t __a, int8x16_t __b);//_mm_andnot_si128
	int16x8_t vbicq_s16 (int16x8_t __a, int16x8_t __b);//_mm_andnot_si128
	int32x4_t vbicq_s32 (int32x4_t __a, int32x4_t __b);//_mm_andnot_si128
	int64x2_t vbicq_s64 (int64x2_t __a, int64x2_t __b);//_mm_andnot_si128
	uint8x16_t vbicq_u8 (uint8x16_t __a, uint8x16_t __b);//_mm_andnot_si128
	uint16x8_t vbicq_u16 (uint16x8_t __a, uint16x8_t __b);//_mm_andnot_si128
	uint32x4_t vbicq_u32 (uint32x4_t __a, uint32x4_t __b);//_mm_andnot_si128
	uint64x2_t vbicq_u64 (uint64x2_t __a, uint64x2_t __b);//_mm_andnot_si128
	/*--6、Bitwise OR complement(正常指令): vorn -> ri = ai | (~bi); 
	performs a bitwise logical OR NOT operation 
	between values in two registers, and places the results in the destination register.--*/
	int8x8_t vorn_s8 (int8x8_t __a, int8x8_t __b);
	int16x4_t vorn_s16 (int16x4_t __a, int16x4_t __b);
	int32x2_t vorn_s32 (int32x2_t __a, int32x2_t __b);
	uint8x8_t vorn_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4_t vorn_u16 (uint16x4_t __a, uint16x4_t __b);
	uint32x2_t vorn_u32 (uint32x2_t __a, uint32x2_t __b);
	int64x1_t vorn_s64 (int64x1_t __a, int64x1_t __b);
	uint64x1_t vorn_u64 (uint64x1_t __a, uint64x1_t __b);
	int8x16_t vornq_s8 (int8x16_t __a, int8x16_t __b);
	int16x8_t vornq_s16 (int16x8_t __a, int16x8_t __b);
	int32x4_t vornq_s32 (int32x4_t __a, int32x4_t __b);
	int64x2_t vornq_s64 (int64x2_t __a, int64x2_t __b);
	uint8x16_t vornq_u8 (uint8x16_t __a, uint8x16_t __b);
	uint16x8_t vornq_u16 (uint16x8_t __a, uint16x8_t __b);
	uint32x4_t vornq_u32 (uint32x4_t __a, uint32x4_t __b);
	uint64x2_t vornq_u64 (uint64x2_t __a, uint64x2_t __b);
	/****************************************Count leading sign bits************************/
	/*--正常指令, vcls -> ; counts the number of consecutive bits, starting from the most 
	significant bit,that are the same as the most significant bit, in each element in a 
	vector, and places the count in the result vector.--*/
	int8x8_t vcls_s8 (int8x8_t __a);
	int16x4_t vcls_s16 (int16x4_t __a);
	int32x2_t vcls_s32 (int32x2_t __a);
	int8x16_t vclsq_s8 (int8x16_t __a);
	int16x8_t vclsq_s16 (int16x8_t __a);
	int32x4_t vclsq_s32 (int32x4_t __a);
	/*******************************************Count leading zeros*************************/
	/*--正常指令, vclz -> ; counts the number of consecutive zeros, starting from the most
	significant bit, in each element in a vector, and places the count in result vector.--*/
	int8x8_t vclz_s8 (int8x8_t __a);
	int16x4_t vclz_s16 (int16x4_t __a);
	int32x2_t vclz_s32 (int32x2_t __a);
	uint8x8_t vclz_u8 (uint8x8_t __a);
	uint16x4_t vclz_u16 (uint16x4_t __a);
	uint32x2_t vclz_u32 (uint32x2_t __a);
	int8x16_t vclzq_s8 (int8x16_t __a);
	int16x8_t vclzq_s16 (int16x8_t __a);
	int32x4_t vclzq_s32 (int32x4_t __a);
	uint8x16_t vclzq_u8 (uint8x16_t __a);
	uint16x8_t vclzq_u16 (uint16x8_t __a);
	uint32x4_t vclzq_u32 (uint32x4_t __a);
	/*******************************************Count number of set bits********************/
	/*--正常指令, vcnt -> counts the number of bits that are one in each element in a vector, 
	and places the count in the result vector.--*/
	int8x8_t vcnt_s8 (int8x8_t __a);
	uint8x8_t vcnt_u8 (uint8x8_t __a);
	poly8x8_t vcnt_p8 (poly8x8_t __a);
	int8x16_t vcntq_s8 (int8x16_t __a);
	uint8x16_t vcntq_u8 (uint8x16_t __a);
	poly8x16_t vcntq_p8 (poly8x16_t __a);
	/*****************************************Reciprocal estimate***************************/
	/*--正常指令, vrecpe -> ; finds an approximate reciprocal of each element in a vector, 
	and places it in the result vector.--*/
	float32x2_t vrecpe_f32 (float32x2_t __a);
	uint32x2_t vrecpe_u32 (uint32x2_t __a);
	float32x4_t vrecpeq_f32 (float32x4_t __a);
	uint32x4_t vrecpeq_u32 (uint32x4_t __a);
	/****************************************Reciprocal square-root estimate****************/
	/*--正常指令, vrsqrte -> ; finds an approximate reciprocal square root of each element
	in a vector, and places it in the return vector.--*/
	float32x2_t vrsqrte_f32 (float32x2_t __a);
	uint32x2_t vrsqrte_u32 (uint32x2_t __a);
	float32x4_t vrsqrteq_f32 (float32x4_t __a);
	uint32x4_t vrsqrteq_u32 (uint32x4_t __a);
	/*******************************************Get lanes from a vector*********************/
	/*--vmov -> r = a[b]; returns the value from the specified lane of a vector.
	Extract lanes from a vector and put into a register. 
	These intrinsics extract a single lane (element) from a vector.--*/
	int8_t vget_lane_s8 (int8x8_t __a, const int __b);//_mm_extract_epi8
	int16_t vget_lane_s16 (int16x4_t __a, const int __b);//_mm_extract_epi16
	int32_t vget_lane_s32 (int32x2_t __a, const int __b);//_mm_extract_epi32
	float32_t vget_lane_f32 (float32x2_t __a, const int __b);
	uint8_t vget_lane_u8 (uint8x8_t __a, const int __b);//_mm_extract_epi8
	uint16_t vget_lane_u16 (uint16x4_t __a, const int __b);//_mm_extract_epi16
	uint32_t vget_lane_u32 (uint32x2_t __a, const int __b);//_mm_extract_epi32
	poly8_t vget_lane_p8 (poly8x8_t __a, const int __b);//_mm_extract_epi8
	poly16_t vget_lane_p16 (poly16x4_t __a, const int __b);//_mm_extract_epi16
	int64_t vget_lane_s64 (int64x1_t __a, const int __b);//_mm_extract_epi64
	uint64_t vget_lane_u64 (uint64x1_t __a, const int __b);//_mm_extract_epi64
	int8_t vgetq_lane_s8 (int8x16_t __a, const int __b);//_mm_extract_epi8
	int16_t vgetq_lane_s16 (int16x8_t __a, const int __b);//_mm_extract_epi16
	int32_t vgetq_lane_s32 (int32x4_t __a, const int __b);//_mm_extract_epi32
	float32_t vgetq_lane_f32 (float32x4_t __a, const int __b);
	uint8_t vgetq_lane_u8 (uint8x16_t __a, const int __b);//_mm_extract_epi8
	uint16_t vgetq_lane_u16 (uint16x8_t __a, const int __b);//_mm_extract_epi16
	uint32_t vgetq_lane_u32 (uint32x4_t __a, const int __b);//_mm_extract_epi32
	poly8_t vgetq_lane_p8 (poly8x16_t __a, const int __b);//_mm_extract_epi8
	poly16_t vgetq_lane_p16 (poly16x8_t __a, const int __b);//_mm_extract_epi16
	int64_t vgetq_lane_s64 (int64x2_t __a, const int __b);//_mm_extract_epi64
	uint64_t vgetq_lane_u64 (uint64x2_t __a, const int __b);//_mm_extract_epi64
	/*********************************************Set lanes in a vector********************/
	/*--vmov -> ; sets the value of the specified lane of a vector. It returns the vector 
	with the new value.Load a single lane of a vector from a literal. These intrinsics set 
	a single lane (element) within a vector.--*/
	int8x8_t vset_lane_s8 (int8_t __a, int8x8_t __b, const int __c);
	int16x4_t vset_lane_s16 (int16_t __a, int16x4_t __b, const int __c);
	int32x2_t vset_lane_s32 (int32_t __a, int32x2_t __b, const int __c);
	float32x2_t vset_lane_f32 (float32_t __a, float32x2_t __b, const int __c);
	uint8x8_t vset_lane_u8 (uint8_t __a, uint8x8_t __b, const int __c);
	uint16x4_t vset_lane_u16 (uint16_t __a, uint16x4_t __b, const int __c);
	uint32x2_t vset_lane_u32 (uint32_t __a, uint32x2_t __b, const int __c);
	poly8x8_t vset_lane_p8 (poly8_t __a, poly8x8_t __b, const int __c);
	poly16x4_t vset_lane_p16 (poly16_t __a, poly16x4_t __b, const int __c);
	int64x1_t vset_lane_s64 (int64_t __a, int64x1_t __b, const int __c);
	uint64x1_t vset_lane_u64 (uint64_t __a, uint64x1_t __b, const int __c);
	int8x16_t vsetq_lane_s8 (int8_t __a, int8x16_t __b, const int __c);
	int16x8_t vsetq_lane_s16 (int16_t __a, int16x8_t __b, const int __c);
	int32x4_t vsetq_lane_s32 (int32_t __a, int32x4_t __b, const int __c);
	float32x4_t vsetq_lane_f32 (float32_t __a, float32x4_t __b, const int __c);
	uint8x16_t vsetq_lane_u8 (uint8_t __a, uint8x16_t __b, const int __c);
	uint16x8_t vsetq_lane_u16 (uint16_t __a, uint16x8_t __b, const int __c);
	uint32x4_t vsetq_lane_u32 (uint32_t __a, uint32x4_t __b, const int __c);
	poly8x16_t vsetq_lane_p8 (poly8_t __a, poly8x16_t __b, const int __c);
	poly16x8_t vsetq_lane_p16 (poly16_t __a, poly16x8_t __b, const int __c);
	int64x2_t vsetq_lane_s64 (int64_t __a, int64x2_t __b, const int __c);
	uint64x2_t vsetq_lane_u64 (uint64_t __a, uint64x2_t __b, const int __c);
	/****************************************Create vector from literal bit pattern*********/
	/*--vmov -> ; creates a vector from a 64-bit pattern. 
	Initialize a vector from a literal bit pattern.--*/
	int8x8_t vcreate_s8 (uint64_t __a);//_mm_loadl_epi64
	int16x4_t vcreate_s16 (uint64_t __a);//_mm_loadl_epi64
	int32x2_t vcreate_s32 (uint64_t __a);//_mm_loadl_epi64
	int64x1_t vcreate_s64 (uint64_t __a);//_mm_loadl_epi64
	float32x2_t vcreate_f32 (uint64_t __a);
	uint8x8_t vcreate_u8 (uint64_t __a);//_mm_loadl_epi64
	uint16x4_t vcreate_u16 (uint64_t __a);//_mm_loadl_epi64
	uint32x2_t vcreate_u32 (uint64_t __a);//_mm_loadl_epi64
	uint64x1_t vcreate_u64 (uint64_t __a);//_mm_loadl_epi64
	poly8x8_t vcreate_p8 (uint64_t __a);//_mm_loadl_epi64
	poly16x4_t vcreate_p16 (uint64_t __a);//_mm_loadl_epi64
	/*****************************************Set all lanes to the same value***************/
	/*--1、Load all lanes of vector to the same literal value: vdup/vmov -> ri = a; 
	duplicates a scalar into every element of the destination vector. 
	Load all lanes of vector to the same literal value--*/
	int8x8_t vdup_n_s8 (int8_t __a);//_mm_set1_epi8
	int16x4_t vdup_n_s16 (int16_t __a);//_mm_set1_epi16
	int32x2_t vdup_n_s32 (int32_t __a);//_mm_set1_epi32
	float32x2_t vdup_n_f32 (float32_t __a);//_mm_set1_ps
	uint8x8_t vdup_n_u8 (uint8_t __a);//_mm_set1_epi8
	uint16x4_t vdup_n_u16 (uint16_t __a);//_mm_set1_epi16
	uint32x2_t vdup_n_u32 (uint32_t __a);//_mm_set1_epi32
	poly8x8_t vdup_n_p8 (poly8_t __a);//_mm_set1_epi8
	poly16x4_t vdup_n_p16 (poly16_t __a);//_mm_set1_epi16
	int64x1_t vdup_n_s64 (int64_t __a);
	uint64x1_t vdup_n_u64 (uint64_t __a);
	int8x16_t vdupq_n_s8 (int8_t __a);//_mm_set1_epi8
	int16x8_t vdupq_n_s16 (int16_t __a);//_mm_set1_epi16
	int32x4_t vdupq_n_s32 (int32_t __a);//_mm_set1_epi32
	float32x4_t vdupq_n_f32 (float32_t __a);//_mm_set1_ps
	uint8x16_t vdupq_n_u8 (uint8_t __a);//_mm_set1_epi8
	uint16x8_t vdupq_n_u16 (uint16_t __a);//_mm_set1_epi16
	uint32x4_t vdupq_n_u32 (uint32_t __a);//_mm_set1_epi32
	poly8x16_t vdupq_n_p8 (poly8_t __a);//_mm_set1_epi8
	poly16x8_t vdupq_n_p16 (poly16_t __a);//_mm_set1_epi16
	int64x2_t vdupq_n_s64 (int64_t __a);
	uint64x2_t vdupq_n_u64 (uint64_t __a);
	int8x8_t vmov_n_s8 (int8_t __a);//_mm_set1_epi8
	int16x4_t vmov_n_s16 (int16_t __a);//_mm_set1_epi16
	int32x2_t vmov_n_s32 (int32_t __a);//_mm_set1_epi32
	float32x2_t vmov_n_f32 (float32_t __a);//_mm_set1_ps
	uint8x8_t vmov_n_u8 (uint8_t __a);//_mm_set1_epi8
	uint16x4_t vmov_n_u16 (uint16_t __a);//_mm_set1_epi16
	uint32x2_t vmov_n_u32 (uint32_t __a);//_mm_set1_epi32
	poly8x8_t vmov_n_p8 (poly8_t __a);//_mm_set1_epi8
	poly16x4_t vmov_n_p16 (poly16_t __a);//_mm_set1_epi16
	int64x1_t vmov_n_s64 (int64_t __a);
	uint64x1_t vmov_n_u64 (uint64_t __a);
	int8x16_t vmovq_n_s8 (int8_t __a);//_mm_set1_epi8
	int16x8_t vmovq_n_s16 (int16_t __a);//_mm_set1_epi16
	int32x4_t vmovq_n_s32 (int32_t __a);//_mm_set1_epi32
	float32x4_t vmovq_n_f32 (float32_t __a);//_mm_set1_ps
	uint8x16_t vmovq_n_u8 (uint8_t __a);//_mm_set1_epi8
	uint16x8_t vmovq_n_u16 (uint16_t __a);//_mm_set1_epi16
	uint32x4_t vmovq_n_u32 (uint32_t __a);//_mm_set1_epi32
	poly8x16_t vmovq_n_p8 (poly8_t __a);//_mm_set1_epi8
	poly16x8_t vmovq_n_p16 (poly16_t __a);//_mm_set1_epi16
	int64x2_t vmovq_n_s64 (int64_t __a);
	uint64x2_t vmovq_n_u64 (uint64_t __a);
	/*--2、Load all lanes of the vector to the value of a lane of a vector: 
	vdup/vmov -> ri = a[b];
	duplicates a scalar into every element of the destination vector.--*/
	int8x8_t vdup_lane_s8 (int8x8_t __a, const int __b);
	int16x4_t vdup_lane_s16 (int16x4_t __a, const int __b);
	int32x2_t vdup_lane_s32 (int32x2_t __a, const int __b);
	float32x2_t vdup_lane_f32 (float32x2_t __a, const int __b);
	uint8x8_t vdup_lane_u8 (uint8x8_t __a, const int __b);
	uint16x4_t vdup_lane_u16 (uint16x4_t __a, const int __b);
	uint32x2_t vdup_lane_u32 (uint32x2_t __a, const int __b);
	poly8x8_t vdup_lane_p8 (poly8x8_t __a, const int __b);
	poly16x4_t vdup_lane_p16 (poly16x4_t __a, const int __b);
	int64x1_t vdup_lane_s64 (int64x1_t __a, const int __b);
	uint64x1_t vdup_lane_u64 (uint64x1_t __a, const int __b);
	int8x16_t vdupq_lane_s8 (int8x8_t __a, const int __b);
	int16x8_t vdupq_lane_s16 (int16x4_t __a, const int __b);
	int32x4_t vdupq_lane_s32 (int32x2_t __a, const int __b);
	float32x4_t vdupq_lane_f32 (float32x2_t __a, const int __b);
	uint8x16_t vdupq_lane_u8 (uint8x8_t __a, const int __b);
	uint16x8_t vdupq_lane_u16 (uint16x4_t __a, const int __b);
	uint32x4_t vdupq_lane_u32 (uint32x2_t __a, const int __b);
	poly8x16_t vdupq_lane_p8 (poly8x8_t __a, const int __b);
	poly16x8_t vdupq_lane_p16 (poly16x4_t __a, const int __b);
	int64x2_t vdupq_lane_s64 (int64x1_t __a, const int __b);//_mm_unpacklo_epi64
	uint64x2_t vdupq_lane_u64 (uint64x1_t __a, const int __b);//_mm_unpacklo_epi64
	/********************************************Combining vectors**************************/
	/*--长指令, -> r0 = a0, ..., r7 = a7, r8 = b0, ..., r15 = b7;
	joins two 64-bit vectors into a single 128-bit vector. 
	The output vector contains twice the number of elements as each input vector. 
	The lower half of the output vector contains the elements of the first input vector.--*/
	int8x16_t vcombine_s8 (int8x8_t __a, int8x8_t __b);//_mm_unpacklo_epi64
	int16x8_t vcombine_s16 (int16x4_t __a, int16x4_t __b);//_mm_unpacklo_epi64
	int32x4_t vcombine_s32 (int32x2_t __a, int32x2_t __b);//_mm_unpacklo_epi64
	int64x2_t vcombine_s64 (int64x1_t __a, int64x1_t __b);//_mm_unpacklo_epi64
	float32x4_t vcombine_f32 (float32x2_t __a, float32x2_t __b);
	uint8x16_t vcombine_u8 (uint8x8_t __a, uint8x8_t __b);//_mm_unpacklo_epi64
	uint16x8_t vcombine_u16 (uint16x4_t __a, uint16x4_t __b);//_mm_unpacklo_epi64
	uint32x4_t vcombine_u32 (uint32x2_t __a, uint32x2_t __b);//_mm_unpacklo_epi64
	uint64x2_t vcombine_u64 (uint64x1_t __a, uint64x1_t __b);//_mm_unpacklo_epi64
	poly8x16_t vcombine_p8 (poly8x8_t __a, poly8x8_t __b);//_mm_unpacklo_epi64
	poly16x8_t vcombine_p16 (poly16x4_t __a, poly16x4_t __b);//_mm_unpacklo_epi64
	/***************************************Splitting vectors*******************************/
	/*--1、窄指令, -> ri = a(i+4); returns the higher half of the 128-bit input vector. The
	output is a 64-bit vector that has half the number of elements as the input vector.--*/
	int8x8_t vget_high_s8 (int8x16_t __a);//_mm_unpackhi_epi64
	int16x4_t vget_high_s16 (int16x8_t __a);//_mm_unpackhi_epi64
	int32x2_t vget_high_s32 (int32x4_t __a);//_mm_unpackhi_epi64
	int64x1_t vget_high_s64 (int64x2_t __a);//_mm_unpackhi_epi64
	float32x2_t vget_high_f32 (float32x4_t __a);
	uint8x8_t vget_high_u8 (uint8x16_t __a);//_mm_unpackhi_epi64
	uint16x4_t vget_high_u16 (uint16x8_t __a);//_mm_unpackhi_epi64
	uint32x2_t vget_high_u32 (uint32x4_t __a);//_mm_unpackhi_epi64
	uint64x1_t vget_high_u64 (uint64x2_t __a);//_mm_unpackhi_epi64
	poly8x8_t vget_high_p8 (poly8x16_t __a);//_mm_unpackhi_epi64
	poly16x4_t vget_high_p16 (poly16x8_t __a);//_mm_unpackhi_epi64
	/*--2、窄指令, -> ri = ai; returns the lower half of the 128-bit input vector. The
	output is a 64-bit vector that has half the number of elements as the input vector.--*/
	int8x8_t vget_low_s8 (int8x16_t __a);
	int16x4_t vget_low_s16 (int16x8_t __a);
	int32x2_t vget_low_s32 (int32x4_t __a);
	float32x2_t vget_low_f32 (float32x4_t __a);
	uint8x8_t vget_low_u8 (uint8x16_t __a);
	uint16x4_t vget_low_u16 (uint16x8_t __a);
	uint32x2_t vget_low_u32 (uint32x4_t __a);
	poly8x8_t vget_low_p8 (poly8x16_t __a);
	poly16x4_t vget_low_p16 (poly16x8_t __a);
	int64x1_t vget_low_s64 (int64x2_t __a);
	uint64x1_t vget_low_u64 (uint64x2_t __a);
	/****************************************************Conversions************************/
	/*--1、Convert from float: vcvt ->, convert from floating-point to integer.--*/
	int32x2_t vcvt_s32_f32 (float32x2_t __a);
	uint32x2_t vcvt_u32_f32 (float32x2_t __a);
	int32x4_t vcvtq_s32_f32 (float32x4_t __a);
	uint32x4_t vcvtq_u32_f32 (float32x4_t __a);
	int32x2_t vcvt_n_s32_f32 (float32x2_t __a, const int __b);
	uint32x2_t vcvt_n_u32_f32 (float32x2_t __a, const int __b);
	int32x4_t vcvtq_n_s32_f32 (float32x4_t __a, const int __b);
	uint32x4_t vcvtq_n_u32_f32 (float32x4_t __a, const int __b);
	/*--2、Convert to float: vcvt ->, convert from integer to floating-point.--*/
	float32x2_t vcvt_f32_s32 (int32x2_t __a);
	float32x2_t vcvt_f32_u32 (uint32x2_t __a);
	float32x4_t vcvtq_f32_s32 (int32x4_t __a);
	float32x4_t vcvtq_f32_u32 (uint32x4_t __a);
	float32x2_t vcvt_n_f32_s32 (int32x2_t __a, const int __b);
	float32x2_t vcvt_n_f32_u32 (uint32x2_t __a, const int __b);
	float32x4_t vcvtq_n_f32_s32 (int32x4_t __a, const int __b);
	float32x4_t vcvtq_n_f32_u32 (uint32x4_t __a, const int __b);
	/*--3、between single-precision and double-precision numbers: vcvt ->--*/
	float16x4_t vcvt_f16_f32(float32x4_t a);
	float32x4_t vcvt_f32_f16(float16x4_t a);
	/*************************************************Move**********************************/
	/*--1、Vector narrow integer(窄指令): vmovn -> ri = ai[0...8]; copies the least 
	significant half of each element of a quadword vector into 
	the corresponding elements of a doubleword vector.--*/
	int8x8_t vmovn_s16 (int16x8_t __a);
	int16x4_t vmovn_s32 (int32x4_t __a);
	int32x2_t vmovn_s64 (int64x2_t __a);
	uint8x8_t vmovn_u16 (uint16x8_t __a);
	uint16x4_t vmovn_u32 (uint32x4_t __a);
	uint32x2_t vmovn_u64 (uint64x2_t __a);
	/*--2、Vector long move(长指令): vmovl -> sign extends or zero extends each element
	in a doubleword vector to twice its original length,
	and places the results in a quadword vector.--*/
	int16x8_t vmovl_s8 (int8x8_t __a);//_mm_cvtepi8_epi16
	int32x4_t vmovl_s16 (int16x4_t __a);//_mm_cvtepi16_epi32
	int64x2_t vmovl_s32 (int32x2_t __a);//_mm_cvtepi32_epi64
	uint16x8_t vmovl_u8 (uint8x8_t __a);//_mm_cvtepu8_epi16
	uint32x4_t vmovl_u16 (uint16x4_t __a);//_mm_cvtepu16_epi32
	uint64x2_t vmovl_u32 (uint32x2_t __a);////_mm_cvtepu32_epi64
	/*--3、Vector saturating narrow integer(窄指令): vqmovn -> copies each element of the
	operand vector to the corresponding element of the destination vector. 
	The result element is half the width of 
	the operand element, and values are saturated to the result width.
	The results are the same type as the operands.--*/
	int8x8_t vqmovn_s16 (int16x8_t __a);//_mm_packs_epi16
	int16x4_t vqmovn_s32 (int32x4_t __a);//_mm_packs_epi32
	int32x2_t vqmovn_s64 (int64x2_t __a);
	uint8x8_t vqmovn_u16 (uint16x8_t __a);
	uint16x4_t vqmovn_u32 (uint32x4_t __a);
	uint32x2_t vqmovn_u64 (uint64x2_t __a);
	/*--4、Vector saturating narrow integer signed->unsigned(窄指令): copies each element of
	the operand vector to the corresponding element of the destination vector.
	The result element is half the width of the operand element,
	and values are saturated to the result width.
	The elements in the operand are signed and the elements in the result are unsigned.--*/
	uint8x8_t vqmovun_s16 (int16x8_t __a);//_mm_packus_epi16
	uint16x4_t vqmovun_s32 (int32x4_t __a);//_mm_packus_epi32
	uint32x2_t vqmovun_s64 (int64x2_t __a);
	/******************************************************Table lookup*********************/
	/*--1、Table lookup: vtbl -> uses byte indexes in a control vector to look up byte 
	values in a table and generate a new vector. Indexes out of range return 0. 
	The table is in Vector1 and uses one(or two or three or four)D registers.--*/
	int8x8_t vtbl1_s8 (int8x8_t __a, int8x8_t __b);
	uint8x8_t vtbl1_u8 (uint8x8_t __a, uint8x8_t __b);
	poly8x8_t vtbl1_p8 (poly8x8_t __a, uint8x8_t __b);
	int8x8_t vtbl2_s8 (int8x8x2_t __a, int8x8_t __b);
	uint8x8_t vtbl2_u8 (uint8x8x2_t __a, uint8x8_t __b);
	poly8x8_t vtbl2_p8 (poly8x8x2_t __a, uint8x8_t __b);
	int8x8_t vtbl3_s8 (int8x8x3_t __a, int8x8_t __b);
	uint8x8_t vtbl3_u8 (uint8x8x3_t __a, uint8x8_t __b);
	poly8x8_t vtbl3_p8 (poly8x8x3_t __a, uint8x8_t __b);
	int8x8_t vtbl4_s8 (int8x8x4_t __a, int8x8_t __b);
	uint8x8_t vtbl4_u8 (uint8x8x4_t __a, uint8x8_t __b);
	poly8x8_t vtbl4_p8 (poly8x8x4_t __a, uint8x8_t __b);
	/*--2、Extended table lookup: vtbx -> uses byte indexes in a control vector to look up
	byte values in a table and generate a new vector. Indexes out of range leave the 
	destination element unchanged.The table is in Vector2 and uses one(or two or three or
	four) D register. Vector1 contains the elements of the destination vector.--*/
	int8x8_t vtbx1_s8 (int8x8_t __a, int8x8_t __b, int8x8_t __c);
	uint8x8_t vtbx1_u8 (uint8x8_t __a, uint8x8_t __b, uint8x8_t __c);
	poly8x8_t vtbx1_p8 (poly8x8_t __a, poly8x8_t __b, uint8x8_t __c);
	int8x8_t vtbx2_s8 (int8x8_t __a, int8x8x2_t __b, int8x8_t __c);
	uint8x8_t vtbx2_u8 (uint8x8_t __a, uint8x8x2_t __b, uint8x8_t __c);
	poly8x8_t vtbx2_p8 (poly8x8_t __a, poly8x8x2_t __b, uint8x8_t __c);
	int8x8_t vtbx3_s8 (int8x8_t __a, int8x8x3_t __b, int8x8_t __c);
	uint8x8_t vtbx3_u8 (uint8x8_t __a, uint8x8x3_t __b, uint8x8_t __c);
	poly8x8_t vtbx3_p8 (poly8x8_t __a, poly8x8x3_t __b, uint8x8_t __c);
	int8x8_t vtbx4_s8 (int8x8_t __a, int8x8x4_t __b, int8x8_t __c);
	uint8x8_t vtbx4_u8 (uint8x8_t __a, uint8x8x4_t __b, uint8x8_t __c);
	poly8x8_t vtbx4_p8 (poly8x8_t __a, poly8x8x4_t __b, uint8x8_t __c);
	/***************************************Multiply, scalar, lane**************************/
	/*--1、Vector multiply by scalar: vmul -> ri = ai * b; 
	multiplies each element in a vector by a scalar, 
	and places the results in the destination vector.--*/
	int16x4_t vmul_n_s16 (int16x4_t __a, int16_t __b);
	int32x2_t vmul_n_s32 (int32x2_t __a, int32_t __b);
	float32x2_t vmul_n_f32 (float32x2_t __a, float32_t __b);
	uint16x4_t vmul_n_u16 (uint16x4_t __a, uint16_t __b);
	uint32x2_t vmul_n_u32 (uint32x2_t __a, uint32_t __b);
	int16x8_t vmulq_n_s16 (int16x8_t __a, int16_t __b);
	int32x4_t vmulq_n_s32 (int32x4_t __a, int32_t __b);
	float32x4_t vmulq_n_f32 (float32x4_t __a, float32_t __b);
	uint16x8_t vmulq_n_u16 (uint16x8_t __a, uint16_t __b);
	uint32x4_t vmulq_n_u32 (uint32x4_t __a, uint32_t __b);
	/*--2、Vector multiply by scalar: -> ri = ai * b[c]; 
	multiplies the first vector by a scalar. 
	The scalar is the element in the second vector with index c.--*/
	int16x4_t vmul_lane_s16 (int16x4_t __a, int16x4_t __b, const int __c);
	int32x2_t vmul_lane_s32 (int32x2_t __a, int32x2_t __b, const int __c);
	float32x2_t vmul_lane_f32 (float32x2_t __a, float32x2_t __b, const int __c);
	uint16x4_t vmul_lane_u16 (uint16x4_t __a, uint16x4_t __b, const int __c);
	uint32x2_t vmul_lane_u32 (uint32x2_t __a, uint32x2_t __b, const int __c);
	int16x8_t vmulq_lane_s16 (int16x8_t __a, int16x4_t __b, const int __c);
	int32x4_t vmulq_lane_s32 (int32x4_t __a, int32x2_t __b, const int __c);
	float32x4_t vmulq_lane_f32 (float32x4_t __a, float32x2_t __b, const int __c);
	uint16x8_t vmulq_lane_u16 (uint16x8_t __a, uint16x4_t __b, const int __c);
	uint32x4_t vmulq_lane_u32 (uint32x4_t __a, uint32x2_t __b, const int __c);
	/*--3、Vector long multiply with scalar: vmull ->  ri = ai * b;
	multiplies a vector by a scalar. 
	Elements in the result are wider than elements in input vector.--*/
	int32x4_t vmull_n_s16 (int16x4_t __a, int16_t __b);
	int64x2_t vmull_n_s32 (int32x2_t __a, int32_t __b);
	uint32x4_t vmull_n_u16 (uint16x4_t __a, uint16_t __b);
	uint64x2_t vmull_n_u32 (uint32x2_t __a, uint32_t __b);
	/*--4、Vector long multiply by scalar: vmull -> ri = ai * b[c];
	multiplies the first vector by a scalar. 
	The scalar is the element in the second vector with index c. 
	The elements in the result are wider than the elements in input vector.--*/
	int32x4_t vmull_lane_s16 (int16x4_t __a, int16x4_t __b, const int __c);
	int64x2_t vmull_lane_s32 (int32x2_t __a, int32x2_t __b, const int __c);
	uint32x4_t vmull_lane_u16 (uint16x4_t __a, uint16x4_t __b, const int __c);
	uint64x2_t vmull_lane_u32 (uint32x2_t __a, uint32x2_t __b, const int __c);
	/*--5、Vector saturating doubling long multiply with scalar: vqdmull -> ri = sat(ai * b);
	multiplies the elements in the vector by a scalar, and doubles the results. 
	If any of the results overflow, they are saturated and the sticky QC flag is set.--*/
	int32x4_t vqdmull_n_s16 (int16x4_t __a, int16_t __b);
	int64x2_t vqdmull_n_s32 (int32x2_t __a, int32_t __b);
	/*--6、Vector saturating doubling long multiply by scalar: vqdmull -> ri = sat(ai * b[c]);
	multiplies the elements in the first vector by a scalar, and doubles the results. 
	The scalar has index c in the second vector. If any of the results overflow, 
	they are saturated and the sticky QC flagis set.--*/
	int32x4_t vqdmull_lane_s16 (int16x4_t __a, int16x4_t __b, const int __c);
	int64x2_t vqdmull_lane_s32 (int32x2_t __a, int32x2_t __b, const int __c);
	/*--7、Vector saturating doubling multiply high with scalar: vqdmulh -> ri = sat(ai * b)
	multiplies the elements of the vector by a scalar, and doubles the results.
	It then returns only the high half of the results.
	If any of the results overflow, they are saturated and the sticky QC flag is set.--*/
	int16x4_t vqdmulh_n_s16 (int16x4_t __a, int16_t __b);
	int32x2_t vqdmulh_n_s32 (int32x2_t __a, int32_t __b);
	int16x8_t vqdmulhq_n_s16 (int16x8_t __a, int16_t __b);
	int32x4_t vqdmulhq_n_s32 (int32x4_t __a, int32_t __b);
	/*--8、Vector saturating doubling multiply high by scalar: 
	vqdmulh -> ri = sat(ai * b[c]);
	multiplies the elements of the first vector by a scalar, and doubles the results. It then
	returns only the high half of the results. The scalar has index n in the second vector.
	If any of the results overflow, they are saturated and the sticky QC flag is set.--*/
	int16x4_t vqdmulh_lane_s16 (int16x4_t __a, int16x4_t __b, const int __c);
	int32x2_t vqdmulh_lane_s32 (int32x2_t __a, int32x2_t __b, const int __c);
	int16x8_t vqdmulhq_lane_s16 (int16x8_t __a, int16x4_t __b, const int __c);
	int32x4_t vqdmulhq_lane_s32 (int32x4_t __a, int32x2_t __b, const int __c);
	/*--9、Vector saturating rounding doubling multiply high with scalar: 
	vqqrdmulh -> ri = sat(ai * b);
	multiplies the elements of the vector by a scalar and doubles the results. 
	It then returns only the high half of the rounded results. 
	If any of the results overflow, they are saturated and the sticky QC flag is set.--*/
	int16x4_t vqrdmulh_n_s16 (int16x4_t __a, int16_t __b);
	int32x2_t vqrdmulh_n_s32 (int32x2_t __a, int32_t __b);
	int16x8_t vqrdmulhq_n_s16 (int16x8_t __a, int16_t __b);
	int32x4_t vqrdmulhq_n_s32 (int32x4_t __a, int32_t __b);
	/*--10、Vector rounding saturating doubling multiply high by scalar: 
	vqrdmulh -> ri = sat(ai * b[c]);
	multiplies the elements of the first vector by a scalar and doubles the results.
	It then returns only the high half of the rounded results.
	The scalar has index n in the second vector. If any of the results overflow, 
	they are saturated and the sticky QC flag is set.--*/
	int16x4_t vqrdmulh_lane_s16 (int16x4_t __a, int16x4_t __b, const int __c);
	int32x2_t vqrdmulh_lane_s32 (int32x2_t __a, int32x2_t __b, const int __c);
	int16x8_t vqrdmulhq_lane_s16 (int16x8_t __a, int16x4_t __b, const int __c);
	int32x4_t vqrdmulhq_lane_s32 (int32x4_t __a, int32x2_t __b, const int __c);
	/*--11、Vector multiply accumulate with scalar: vmla -> ri = ai + bi * c;
	multiplies each element in the second vector by a scalar, 
	and adds the results to the corresponding elements of the first vector.--*/
	int16x4_t vmla_n_s16 (int16x4_t __a, int16x4_t __b, int16_t __c);
	int32x2_t vmla_n_s32 (int32x2_t __a, int32x2_t __b, int32_t __c);
	float32x2_t vmla_n_f32 (float32x2_t __a, float32x2_t __b, float32_t __c);
	uint16x4_t vmla_n_u16 (uint16x4_t __a, uint16x4_t __b, uint16_t __c);
	uint32x2_t vmla_n_u32 (uint32x2_t __a, uint32x2_t __b, uint32_t __c);
	int16x8_t vmlaq_n_s16 (int16x8_t __a, int16x8_t __b, int16_t __c);
	int32x4_t vmlaq_n_s32 (int32x4_t __a, int32x4_t __b, int32_t __c);
	float32x4_t vmlaq_n_f32 (float32x4_t __a, float32x4_t __b, float32_t __c);
	uint16x8_t vmlaq_n_u16 (uint16x8_t __a, uint16x8_t __b, uint16_t __c);
	uint32x4_t vmlaq_n_u32 (uint32x4_t __a, uint32x4_t __b, uint32_t __c);
	/*--12、Vector multiply accumulate by scalar: vmla -> ri = ai + bi * c[d];
	multiplies each element in the second vector by a scalar, 
	and adds the results to the corresponding elements of the first vector. 
	The scalar has index d in the third vector.--*/
	int16x4_t vmla_lane_s16 (int16x4_t __a, int16x4_t __b, int16x4_t __c, const int __d);
	int32x2_t vmla_lane_s32 (int32x2_t __a, int32x2_t __b, int32x2_t __c, const int __d);
	float32x2_t vmla_lane_f32 (float32x2_t __a, float32x2_t __b, float32x2_t __c,
		const int __d);
	uint16x4_t vmla_lane_u16 (uint16x4_t __a, uint16x4_t __b, uint16x4_t __c, const int __d);
	uint32x2_t vmla_lane_u32 (uint32x2_t __a, uint32x2_t __b, uint32x2_t __c, const int __d);
	int16x8_t vmlaq_lane_s16 (int16x8_t __a, int16x8_t __b, int16x4_t __c, const int __d);
	int32x4_t vmlaq_lane_s32 (int32x4_t __a, int32x4_t __b, int32x2_t __c, const int __d);
	float32x4_t vmlaq_lane_f32 (float32x4_t __a, float32x4_t __b, float32x2_t __c,
		const int __d);
	uint16x8_t vmlaq_lane_u16 (uint16x8_t __a, uint16x8_t __b, uint16x4_t __c, const int __d);
	uint32x4_t vmlaq_lane_u32 (uint32x4_t __a, uint32x4_t __b, uint32x2_t __c, const int __d);
	/*--13、Vector widening multiply accumulate with scalar: vmlal -> ri = ai + bi * c;
	multiplies each element in the second vector by a scalar, and adds the results into the 
	corresponding elements of the first vector. 
	The scalar has index n in the third vector. The elements in the result are wider.--*/
	int32x4_t vmlal_n_s16 (int32x4_t __a, int16x4_t __b, int16_t __c);
	int64x2_t vmlal_n_s32 (int64x2_t __a, int32x2_t __b, int32_t __c);
	uint32x4_t vmlal_n_u16 (uint32x4_t __a, uint16x4_t __b, uint16_t __c);
	uint64x2_t vmlal_n_u32 (uint64x2_t __a, uint32x2_t __b, uint32_t __c);
	/*--14、Vector widening multiply accumulate by scalar: vmlal -> ri = ai + bi * c[d];
	multiplies each element in the second vector by a scalar, and adds the results to the 
	corresponding elements of the first vector. The scalar has index d in the third vector.
	The elements in the result are wider.--*/
	int32x4_t vmlal_lane_s16 (int32x4_t __a, int16x4_t __b, int16x4_t __c, const int __d);
	int64x2_t vmlal_lane_s32 (int64x2_t __a, int32x2_t __b, int32x2_t __c, const int __d);
	uint32x4_t vmlal_lane_u16 (uint32x4_t __a, uint16x4_t __b, uint16x4_t __c, const int __d);
	uint64x2_t vmlal_lane_u32 (uint64x2_t __a, uint32x2_t __b, uint32x2_t __c, const int __d);
	/*--15、Vector widening saturating doubling multiply accumulate with scalar: 
	vqdmlal -> ri = sat(ai + bi * c);
	multiplies the elements in the second vector by a scalar, and doubles the results. 
	It then adds the results to the elements in the first vector.
	If any of the results overflow, they are saturated and the sticky QC flag is set.--*/
	int32x4_t vqdmlal_n_s16 (int32x4_t __a, int16x4_t __b, int16_t __c);
	int64x2_t vqdmlal_n_s32 (int64x2_t __a, int32x2_t __b, int32_t __c);
	/*--16、Vector widening saturating doubling multiply accumulate by scalar: 
	vqdmlal -> ri = sat(ai + bi * c[d])
	multiplies each element in the second vector by a scalar, doubles the results and adds 
	them to the corresponding elements of the first vector. The scalar has index d in the 
	third vector. If any of the results overflow,
	they are saturated and the sticky QC flag is set.--*/
	int32x4_t vqdmlal_lane_s16 (int32x4_t __a, int16x4_t __b, int16x4_t __c, const int __d);
	int64x2_t vqdmlal_lane_s32 (int64x2_t __a, int32x2_t __b, int32x2_t __c, const int __d);
	/*--17、Vector multiply subtract with scalar: vmls -> ri = ai - bi * c;
	multiplies each element in a vector by a scalar, subtracts the results from the 
	corresponding elements of the destination vector, 
	and places the final results in the destination vector.--*/
	int16x4_t vmls_n_s16 (int16x4_t __a, int16x4_t __b, int16_t __c);
	int32x2_t vmls_n_s32 (int32x2_t __a, int32x2_t __b, int32_t __c);
	float32x2_t vmls_n_f32 (float32x2_t __a, float32x2_t __b, float32_t __c);
	uint16x4_t vmls_n_u16 (uint16x4_t __a, uint16x4_t __b, uint16_t __c);
	uint32x2_t vmls_n_u32 (uint32x2_t __a, uint32x2_t __b, uint32_t __c);
	int16x8_t vmlsq_n_s16 (int16x8_t __a, int16x8_t __b, int16_t __c);
	int32x4_t vmlsq_n_s32 (int32x4_t __a, int32x4_t __b, int32_t __c);
	float32x4_t vmlsq_n_f32 (float32x4_t __a, float32x4_t __b, float32_t __c);
	uint16x8_t vmlsq_n_u16 (uint16x8_t __a, uint16x8_t __b, uint16_t __c);
	uint32x4_t vmlsq_n_u32 (uint32x4_t __a, uint32x4_t __b, uint32_t __c);
	/*--18、Vector multiply subtract by scalar: vmls -> ri = ai - bi * c[d];
	multiplies each element in the second vector by a scalar, and subtracts them from the
	corresponding elements of the first vector.
	The scalar has index d in the third vector.--*/
	int16x4_t vmls_lane_s16 (int16x4_t __a, int16x4_t __b, int16x4_t __c, const int __d);
	int32x2_t vmls_lane_s32 (int32x2_t __a, int32x2_t __b, int32x2_t __c, const int __d);
	float32x2_t vmls_lane_f32 (float32x2_t __a, float32x2_t __b, float32x2_t __c,
		const int __d);
	uint16x4_t vmls_lane_u16 (uint16x4_t __a, uint16x4_t __b, uint16x4_t __c, const int __d);
	uint32x2_t vmls_lane_u32 (uint32x2_t __a, uint32x2_t __b, uint32x2_t __c, const int __d);
	int16x8_t vmlsq_lane_s16 (int16x8_t __a, int16x8_t __b, int16x4_t __c, const int __d);
	int32x4_t vmlsq_lane_s32 (int32x4_t __a, int32x4_t __b, int32x2_t __c, const int __d);
	float32x4_t vmlsq_lane_f32 (float32x4_t __a, float32x4_t __b, float32x2_t __c,
		const int __d);
	uint16x8_t vmlsq_lane_u16 (uint16x8_t __a, uint16x8_t __b, uint16x4_t __c, const int __d);
	uint32x4_t vmlsq_lane_u32 (uint32x4_t __a, uint32x4_t __b, uint32x2_t __c, const int __d);
	/*--19、Vector widening multiply subtract with scalar: vmlsl -> ri = ai - bi * c;
	multiplies the elements in the second vector by a scalar, then subtracts the results from
	the elements in the first vector. The elements of the result are wider.--*/
	int32x4_t vmlsl_n_s16 (int32x4_t __a, int16x4_t __b, int16_t __c);
	int64x2_t vmlsl_n_s32 (int64x2_t __a, int32x2_t __b, int32_t __c);
	uint32x4_t vmlsl_n_u16 (uint32x4_t __a, uint16x4_t __b, uint16_t __c);
	uint64x2_t vmlsl_n_u32 (uint64x2_t __a, uint32x2_t __b, uint32_t __c);
	/*--20、Vector widening multiply subtract by scalar: vmlsl -> ri = ai - bi * c[d];
	multiplies each element in the second vector by a scalar, 
	and subtracts them from the corresponding elements of the first vector. 
	The scalar has index d in the third vector. The elements in the result are wider.--*/
	int32x4_t vmlsl_lane_s16 (int32x4_t __a, int16x4_t __b, int16x4_t __c, const int __d);
	int64x2_t vmlsl_lane_s32 (int64x2_t __a, int32x2_t __b, int32x2_t __c, const int __d);
	uint32x4_t vmlsl_lane_u16 (uint32x4_t __a, uint16x4_t __b, uint16x4_t __c, const int __d)
		uint64x2_t vmlsl_lane_u32 (uint64x2_t __a, uint32x2_t __b, uint32x2_t __c, const int __d);
	/*--21、Vector widening saturating doubling multiply subtract with scalar: 
	vqdmlsl -> ri = sat(ai - bi * c);
	multiplies the elements of the second vector with a scalar and doubles the results. 
	It then subtracts the results from the elements in the first vector.
	If any of the results overflow, they are saturated and the sticky QC flag is set.--*/
	int32x4_t vqdmlsl_n_s16 (int32x4_t __a, int16x4_t __b, int16_t __c);
	int64x2_t vqdmlsl_n_s32 (int64x2_t __a, int32x2_t __b, int32_t __c);
	/*--22、Vector widening saturating doubling multiply subtract by scalar:
	vqdmlsl -> ri = sat(ai - bi * c[[d]);
	multiplies each element in the second vector by a scalar, doubles the results and subtracts
	them from the corresponding elements of the first vector. The scalar has index n in the 
	third vector.If any of the results overflow, 
	they are saturated and the sticky QC flag is set.--*/
	int32x4_t vqdmlsl_lane_s16 (int32x4_t __a, int16x4_t __b, int16x4_t __c, const int __d);
	int64x2_t vqdmlsl_lane_s32 (int64x2_t __a, int32x2_t __b, int32x2_t __c, const int __d);
	/*****************************************************Vector extract********************/
	/*--Vector extract: vext -> extracts n elements from the lower end of the second operand
	vector and the remaining elements from the higher end of the first, and combines them to
	form the result vector. The elements from the second operand are placed in the most 
	significant part of the result vector.The elements from the first operand are placed in
	the least significant part of the result vector.This intrinsic cycles the elements
	through the lanes if the two input vectors are the same.--*/
	int8x8_t vext_s8 (int8x8_t __a, int8x8_t __b, const int __c);
	int16x4_t vext_s16 (int16x4_t __a, int16x4_t __b, const int __c);
	int32x2_t vext_s32 (int32x2_t __a, int32x2_t __b, const int __c);
	int64x1_t vext_s64 (int64x1_t __a, int64x1_t __b, const int __c);
	float32x2_t vext_f32 (float32x2_t __a, float32x2_t __b, const int __c);
	uint8x8_t vext_u8 (uint8x8_t __a, uint8x8_t __b, const int __c);
	uint16x4_t vext_u16 (uint16x4_t __a, uint16x4_t __b, const int __c);
	uint32x2_t vext_u32 (uint32x2_t __a, uint32x2_t __b, const int __c);
	uint64x1_t vext_u64 (uint64x1_t __a, uint64x1_t __b, const int __c);
	poly8x8_t vext_p8 (poly8x8_t __a, poly8x8_t __b, const int __c);
	poly16x4_t vext_p16 (poly16x4_t __a, poly16x4_t __b, const int __c);
	int8x16_t vextq_s8 (int8x16_t __a, int8x16_t __b, const int __c);//_mm_alignr_epi8 
	int16x8_t vextq_s16 (int16x8_t __a, int16x8_t __b, const int __c);//_mm_alignr_epi8 
	int32x4_t vextq_s32 (int32x4_t __a, int32x4_t __b, const int __c);//_mm_alignr_epi8
	int64x2_t vextq_s64 (int64x2_t __a, int64x2_t __b, const int __c);//_mm_alignr_epi8
	float32x4_t vextq_f32 (float32x4_t __a, float32x4_t __b, const int __c);//_mm_alignr_epi8
	uint8x16_t vextq_u8 (uint8x16_t __a, uint8x16_t __b, const int __c);//_mm_alignr_epi8
	uint16x8_t vextq_u16 (uint16x8_t __a, uint16x8_t __b, const int __c);//_mm_alignr_epi8
	uint32x4_t vextq_u32 (uint32x4_t __a, uint32x4_t __b, const int __c);//_mm_alignr_epi8
	uint64x2_t vextq_u64 (uint64x2_t __a, uint64x2_t __b, const int __c);//_mm_alignr_epi8
	poly8x16_t vextq_p8 (poly8x16_t __a, poly8x16_t __b, const int __c);//_mm_alignr_epi8
	poly16x8_t vextq_p16 (poly16x8_t __a, poly16x8_t __b, const int __c);//_mm_alignr_epi8
	/****************************************************Reverse elements*******************/
	/*--1、Reverse vector elements (swap endianness): vrev64 -> reverses the order of 8-bit, 
	16-bit, or 32-bit elements within each doubleword of the vector, 
	and places the result in the corresponding destination vector.--*/
	int8x8_t vrev64_s8 (int8x8_t __a);
	int16x4_t vrev64_s16 (int16x4_t __a);
	int32x2_t vrev64_s32 (int32x2_t __a);
	float32x2_t vrev64_f32 (float32x2_t __a);//_mm_shuffle_ps
	uint8x8_t vrev64_u8 (uint8x8_t __a);
	uint16x4_t vrev64_u16 (uint16x4_t __a);
	uint32x2_t vrev64_u32 (uint32x2_t __a);
	poly8x8_t vrev64_p8 (poly8x8_t __a);
	poly16x4_t vrev64_p16 (poly16x4_t __a);
	int8x16_t vrev64q_s8 (int8x16_t __a);
	int16x8_t vrev64q_s16 (int16x8_t __a);
	int32x4_t vrev64q_s32 (int32x4_t __a);
	float32x4_t vrev64q_f32 (float32x4_t __a);//_mm_shuffle_ps
	uint8x16_t vrev64q_u8 (uint8x16_t __a);
	uint16x8_t vrev64q_u16 (uint16x8_t __a);
	uint32x4_t vrev64q_u32 (uint32x4_t __a);
	poly8x16_t vrev64q_p8 (poly8x16_t __a);
	poly16x8_t vrev64q_p16 (poly16x8_t __a);
	/*--2、Reverse vector elements (swap endianness): vrev32 -> reverses the order of 8-bit 
	or 16-bit elements within each word of the vector, 
	and places the result in the corresponding destination vector.--*/
	int8x8_t vrev32_s8 (int8x8_t __a);
	int16x4_t vrev32_s16 (int16x4_t __a);
	uint8x8_t vrev32_u8 (uint8x8_t __a);
	uint16x4_t vrev32_u16 (uint16x4_t __a);
	poly8x8_t vrev32_p8 (poly8x8_t __a);
	poly16x4_t vrev32_p16 (poly16x4_t __a);
	int8x16_t vrev32q_s8 (int8x16_t __a);
	int16x8_t vrev32q_s16 (int16x8_t __a);
	uint8x16_t vrev32q_u8 (uint8x16_t __a);
	uint16x8_t vrev32q_u16 (uint16x8_t __a);
	poly8x16_t vrev32q_p8 (poly8x16_t __a);
	poly16x8_t vrev32q_p16 (poly16x8_t __a);
	/*--3、Reverse vector elements (swap endianness): vrev16 -> reverses the order 
	of 8-bit elements within each halfword of the vector, 
	and places the result in the corresponding destination vector.--*/
	int8x8_t vrev16_s8 (int8x8_t __a);
	uint8x8_t vrev16_u8 (uint8x8_t __a);
	poly8x8_t vrev16_p8 (poly8x8_t __a);
	int8x16_t vrev16q_s8 (int8x16_t __a);
	uint8x16_t vrev16q_u8 (uint8x16_t __a);
	poly8x16_t vrev16q_p8 (poly8x16_t __a);
	/**********************************************************Bitwise Select***************/
	/*--Bitwise Select: vbsl -> selects each bit for the destination from the first operand 
	if the corresponding bit of the destination is 1, 
	or from the second operand if the corresponding bit of the destination is 0.--*/
	int8x8_t vbsl_s8 (uint8x8_t __a, int8x8_t __b, int8x8_t __c);
	int16x4_t vbsl_s16 (uint16x4_t __a, int16x4_t __b, int16x4_t __c);
	int32x2_t vbsl_s32 (uint32x2_t __a, int32x2_t __b, int32x2_t __c);
	int64x1_t vbsl_s64 (uint64x1_t __a, int64x1_t __b, int64x1_t __c);
	float32x2_t vbsl_f32 (uint32x2_t __a, float32x2_t __b, float32x2_t __c);
	uint8x8_t vbsl_u8 (uint8x8_t __a, uint8x8_t __b, uint8x8_t __c);
	uint16x4_t vbsl_u16 (uint16x4_t __a, uint16x4_t __b, uint16x4_t __c);
	uint32x2_t vbsl_u32 (uint32x2_t __a, uint32x2_t __b, uint32x2_t __c);
	uint64x1_t vbsl_u64 (uint64x1_t __a, uint64x1_t __b, uint64x1_t __c);
	poly8x8_t vbsl_p8 (uint8x8_t __a, poly8x8_t __b, poly8x8_t __c);
	poly16x4_t vbsl_p16 (uint16x4_t __a, poly16x4_t __b, poly16x4_t __c);
	int8x16_t vbslq_s8 (uint8x16_t __a, int8x16_t __b, int8x16_t __c);
	int16x8_t vbslq_s16 (uint16x8_t __a, int16x8_t __b, int16x8_t __c);
	int32x4_t vbslq_s32 (uint32x4_t __a, int32x4_t __b, int32x4_t __c);
	int64x2_t vbslq_s64 (uint64x2_t __a, int64x2_t __b, int64x2_t __c);
	float32x4_t vbslq_f32 (uint32x4_t __a, float32x4_t __b, float32x4_t __c);
	uint8x16_t vbslq_u8 (uint8x16_t __a, uint8x16_t __b, uint8x16_t __c);
	uint16x8_t vbslq_u16 (uint16x8_t __a, uint16x8_t __b, uint16x8_t __c);
	uint32x4_t vbslq_u32 (uint32x4_t __a, uint32x4_t __b, uint32x4_t __c);
	uint64x2_t vbslq_u64 (uint64x2_t __a, uint64x2_t __b, uint64x2_t __c);
	poly8x16_t vbslq_p8 (uint8x16_t __a, poly8x16_t __b, poly8x16_t __c);
	poly16x8_t vbslq_p16 (uint16x8_t __a, poly16x8_t __b, poly16x8_t __c);
	/************************************Transposition operations***************************/
	/*--1、Transpose elements: vtrn -> treats the elements of its input vectors as elements
	of 2 x 2 matrices, and transposes the matrices. Essentially, it exchanges the elements 
	with odd indices from Vector1 with the elements with even indices from Vector2.--*/
	int8x8x2_t vtrn_s8 (int8x8_t __a, int8x8_t __b);
	int16x4x2_t vtrn_s16 (int16x4_t __a, int16x4_t __b);
	uint8x8x2_t vtrn_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4x2_t vtrn_u16 (uint16x4_t __a, uint16x4_t __b);
	poly8x8x2_t vtrn_p8 (poly8x8_t __a, poly8x8_t __b);
	poly16x4x2_t vtrn_p16 (poly16x4_t __a, poly16x4_t __b);
	int32x2x2_t vtrn_s32 (int32x2_t __a, int32x2_t __b)
	float32x2x2_t vtrn_f32 (float32x2_t __a, float32x2_t __b)
	uint32x2x2_t vtrn_u32 (uint32x2_t __a, uint32x2_t __b)
	int8x16x2_t vtrnq_s8 (int8x16_t __a, int8x16_t __b)
	int16x8x2_t vtrnq_s16 (int16x8_t __a, int16x8_t __b)
	int32x4x2_t vtrnq_s32 (int32x4_t __a, int32x4_t __b)
	float32x4x2_t vtrnq_f32 (float32x4_t __a, float32x4_t __b)
	uint8x16x2_t vtrnq_u8 (uint8x16_t __a, uint8x16_t __b)
	uint16x8x2_t vtrnq_u16 (uint16x8_t __a, uint16x8_t __b)
	uint32x4x2_t vtrnq_u32 (uint32x4_t __a, uint32x4_t __b);
	poly8x16x2_t vtrnq_p8 (poly8x16_t __a, poly8x16_t __b);
	poly16x8x2_t vtrnq_p16 (poly16x8_t __a, poly16x8_t __b);
	/*--2、Interleave elements(Zip elements): 
	vzip ->  (Vector Zip) interleaves the elements of two vectors.--*/
	int8x8x2_t vzip_s8 (int8x8_t __a, int8x8_t __b);
	int16x4x2_t vzip_s16 (int16x4_t __a, int16x4_t __b);
	uint8x8x2_t vzip_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4x2_t vzip_u16 (uint16x4_t __a, uint16x4_t __b);
	poly8x8x2_t vzip_p8 (poly8x8_t __a, poly8x8_t __b);
	poly16x4x2_t vzip_p16 (poly16x4_t __a, poly16x4_t __b);
	int32x2x2_t vzip_s32 (int32x2_t __a, int32x2_t __b);
	float32x2x2_t vzip_f32 (float32x2_t __a, float32x2_t __b);
	uint32x2x2_t vzip_u32 (uint32x2_t __a, uint32x2_t __b);
	int8x16x2_t vzipq_s8 (int8x16_t __a, int8x16_t __b);
	int16x8x2_t vzipq_s16 (int16x8_t __a, int16x8_t __b);
	int32x4x2_t vzipq_s32 (int32x4_t __a, int32x4_t __b);
	float32x4x2_t vzipq_f32 (float32x4_t __a, float32x4_t __b);
	uint8x16x2_t vzipq_u8 (uint8x16_t __a, uint8x16_t __b);
	uint16x8x2_t vzipq_u16 (uint16x8_t __a, uint16x8_t __b);
	uint32x4x2_t vzipq_u32 (uint32x4_t __a, uint32x4_t __b);
	poly8x16x2_t vzipq_p8 (poly8x16_t __a, poly8x16_t __b);
	poly16x8x2_t vzipq_p16 (poly16x8_t __a, poly16x8_t __b);
	/*--3、De-Interleave elements(Unzip elements): 
	vuzp -> (Vector Unzip) de-interleaves the elements of two vectors.
	De-interleaving is the inverse process of interleaving.--*/
	int8x8x2_t vuzp_s8 (int8x8_t __a, int8x8_t __b);
	int16x4x2_t vuzp_s16 (int16x4_t __a, int16x4_t __b);
	int32x2x2_t vuzp_s32 (int32x2_t __a, int32x2_t __b);
	float32x2x2_t vuzp_f32 (float32x2_t __a, float32x2_t __b);
	uint8x8x2_t vuzp_u8 (uint8x8_t __a, uint8x8_t __b);
	uint16x4x2_t vuzp_u16 (uint16x4_t __a, uint16x4_t __b);
	uint32x2x2_t vuzp_u32 (uint32x2_t __a, uint32x2_t __b);
	poly8x8x2_t vuzp_p8 (poly8x8_t __a, poly8x8_t __b);
	poly16x4x2_t vuzp_p16 (poly16x4_t __a, poly16x4_t __b);
	int8x16x2_t vuzpq_s8 (int8x16_t __a, int8x16_t __b);
	int16x8x2_t vuzpq_s16 (int16x8_t __a, int16x8_t __b);
	int32x4x2_t vuzpq_s32 (int32x4_t __a, int32x4_t __b);
	float32x4x2_t vuzpq_f32 (float32x4_t __a, float32x4_t __b);
	uint8x16x2_t vuzpq_u8 (uint8x16_t __a, uint8x16_t __b);
	uint16x8x2_t vuzpq_u16 (uint16x8_t __a, uint16x8_t __b);
	uint32x4x2_t vuzpq_u32 (uint32x4_t __a, uint32x4_t __b);
	poly8x16x2_t vuzpq_p8 (poly8x16_t __a, poly8x16_t __b);
	poly16x8x2_t vuzpq_p16 (poly16x8_t __a, poly16x8_t __b);
	/*********************************************************Load**************************/
	/*--1、Load a single vector from memory: vld1 -> loads a vector from memory.--*/
	int8x8_t vld1_s8 (const int8_t * __a);
	int16x4_t vld1_s16 (const int16_t * __a);
	int32x2_t vld1_s32 (const int32_t * __a);
	int64x1_t vld1_s64 (const int64_t * __a);
	float32x2_t vld1_f32 (const float32_t * __a);
	uint8x8_t vld1_u8 (const uint8_t * __a);//_mm_loadl_epi64
	uint16x4_t vld1_u16 (const uint16_t * __a);//_mm_loadl_epi64
	uint32x2_t vld1_u32 (const uint32_t * __a);//_mm_loadl_epi64
	uint64x1_t vld1_u64 (const uint64_t * __a);//_mm_loadl_epi64
	poly8x8_t vld1_p8 (const poly8_t * __a);
	poly16x4_t vld1_p16 (const poly16_t * __a);
	int8x16_t vld1q_s8 (const int8_t * __a);
	int16x8_t vld1q_s16 (const int16_t * __a);
	int32x4_t vld1q_s32 (const int32_t * __a);
	int64x2_t vld1q_s64 (const int64_t * __a);
	float32x4_t vld1q_f32 (const float32_t * __a);
	uint8x16_t vld1q_u8 (const uint8_t * __a);
	uint16x8_t vld1q_u16 (const uint16_t * __a);
	uint32x4_t vld1q_u32 (const uint32_t * __a);
	uint64x2_t vld1q_u64 (const uint64_t * __a);
	poly8x16_t vld1q_p8 (const poly8_t * __a);
	poly16x8_t vld1q_p16 (const poly16_t * __a);
	/*--2、Load a single lane from memory: vld1 -> loads one element of the input vector 
	from memory and returns this in the result vector. Elements of the vector that are not
	loaded are returned in the result vector unaltered. 
	c is the index of the element to load.--*/
	int8x8_t vld1_lane_s8 (const int8_t * __a, int8x8_t __b, const int __c);//_mm_insert_epi8
	int16x4_t vld1_lane_s16 (const int16_t * __a, int16x4_t __b,
		const int __c);//_mm_insert_epi16
	int32x2_t vld1_lane_s32 (const int32_t * __a, int32x2_t __b, 
		const int __c);//_mm_insert_epi32
	float32x2_t vld1_lane_f32 (const float32_t * __a, float32x2_t __b, const int __c);
	uint8x8_t vld1_lane_u8 (const uint8_t * __a, uint8x8_t __b, 
		const int __c);//_mm_insert_epi8
	uint16x4_t vld1_lane_u16 (const uint16_t * __a, uint16x4_t __b, 
		const int __c);//_mm_insert_epi16
	uint32x2_t vld1_lane_u32 (const uint32_t * __a, uint32x2_t __b, 
		const int __c);//_mm_insert_epi32
	poly8x8_t vld1_lane_p8 (const poly8_t * __a, poly8x8_t __b, 
		const int __c);//_mm_insert_epi8
	poly16x4_t vld1_lane_p16 (const poly16_t * __a, poly16x4_t __b, 
		const int __c);//_mm_insert_epi16
	int64x1_t vld1_lane_s64 (const int64_t * __a, int64x1_t __b, const int __c);
	uint64x1_t vld1_lane_u64 (const uint64_t * __a, uint64x1_t __b, const int __c);
	int8x16_t vld1q_lane_s8 (const int8_t * __a, int8x16_t __b, 
		const int __c);//_mm_insert_epi8
	int16x8_t vld1q_lane_s16 (const int16_t * __a, int16x8_t __b, 
		const int __c);//_mm_insert_epi16
	int32x4_t vld1q_lane_s32 (const int32_t * __a, int32x4_t __b, 
		const int __c);//_mm_insert_epi32
	float32x4_t vld1q_lane_f32 (const float32_t * __a, float32x4_t __b, const int __c);
	uint8x16_t vld1q_lane_u8 (const uint8_t * __a, uint8x16_t __b, 
		const int __c);//_mm_insert_epi8
	uint16x8_t vld1q_lane_u16 (const uint16_t * __a, uint16x8_t __b, 
		const int __c);//_mm_insert_epi16
	uint32x4_t vld1q_lane_u32 (const uint32_t * __a, uint32x4_t __b, 
		const int __c);//_mm_insert_epi32
	poly8x16_t vld1q_lane_p8 (const poly8_t * __a, poly8x16_t __b, 
		const int __c);//_mm_insert_epi8
	poly16x8_t vld1q_lane_p16 (const poly16_t * __a, poly16x8_t __b, 
		const int __c);//_mm_insert_epi16
	int64x2_t vld1q_lane_s64 (const int64_t * __a, int64x2_t __b, 
		const int __c);//_mm_insert_epi64
	uint64x2_t vld1q_lane_u64 (const uint64_t * __a, uint64x2_t __b, 
		const int __c);//_mm_insert_epi64
	/*--3、Load all lanes of vector with same value from memory: vld1 -> 
	loads one element in a vector from memory. 
	The loaded element is copied to all other lanes of the vector.--*/
	int8x8_t vld1_dup_s8 (const int8_t * __a);//_mm_set1_epi8
	int16x4_t vld1_dup_s16 (const int16_t * __a);//_mm_set1_epi16
	int32x2_t vld1_dup_s32 (const int32_t * __a);//_mm_set1_epi32
	float32x2_t vld1_dup_f32 (const float32_t * __a);//_mm_set1_ps
	uint8x8_t vld1_dup_u8 (const uint8_t * __a);//_mm_set1_epi8
	uint16x4_t vld1_dup_u16 (const uint16_t * __a);//_mm_set1_epi16
	uint32x2_t vld1_dup_u32 (const uint32_t * __a);//_mm_set1_epi32
	poly8x8_t vld1_dup_p8 (const poly8_t * __a);//_mm_set1_epi8
	poly16x4_t vld1_dup_p16 (const poly16_t * __a);//_mm_set1_epi16
	int64x1_t vld1_dup_s64 (const int64_t * __a);
	uint64x1_t vld1_dup_u64 (const uint64_t * __a);
	int8x16_t vld1q_dup_s8 (const int8_t * __a);//_mm_set1_epi8
	int16x8_t vld1q_dup_s16 (const int16_t * __a);//_mm_set1_epi16
	int32x4_t vld1q_dup_s32 (const int32_t * __a);//_mm_set1_epi32
	float32x4_t vld1q_dup_f32 (const float32_t * __a);//_mm_set1_ps
	uint8x16_t vld1q_dup_u8 (const uint8_t * __a);//_mm_set1_epi8
	uint16x8_t vld1q_dup_u16 (const uint16_t * __a);//_mm_set1_epi16
	uint32x4_t vld1q_dup_u32 (const uint32_t * __a);//_mm_set1_epi32
	poly8x16_t vld1q_dup_p8 (const poly8_t * __a);//_mm_set1_epi8
	poly16x8_t vld1q_dup_p16 (const poly16_t * __a);//_mm_set1_epi16
	int64x2_t vld1q_dup_s64 (const int64_t * __a);
	uint64x2_t vld1q_dup_u64 (const uint64_t * __a);
	/*--4、Load 2-element structure from memory: vld2 -> loads 2 vectors from memory. 
	It performs a 2-way de-interleave from memory to the vectors.--*/
	int8x8x2_t vld2_s8 (const int8_t * __a);
	int16x4x2_t vld2_s16 (const int16_t * __a);
	int32x2x2_t vld2_s32 (const int32_t * __a);
	float32x2x2_t vld2_f32 (const float32_t * __a);
	uint8x8x2_t vld2_u8 (const uint8_t * __a);
	uint16x4x2_t vld2_u16 (const uint16_t * __a);
	uint32x2x2_t vld2_u32 (const uint32_t * __a);
	poly8x8x2_t vld2_p8 (const poly8_t * __a);
	poly16x4x2_t vld2_p16 (const poly16_t * __a);
	int64x1x2_t vld2_s64 (const int64_t * __a);
	uint64x1x2_t vld2_u64 (const uint64_t * __a);
	int8x16x2_t vld2q_s8 (const int8_t * __a);
	int16x8x2_t vld2q_s16 (const int16_t * __a);
	int32x4x2_t vld2q_s32 (const int32_t * __a);
	float32x4x2_t vld2q_f32 (const float32_t * __a);
	uint8x16x2_t vld2q_u8 (const uint8_t * __a);
	uint16x8x2_t vld2q_u16 (const uint16_t * __a);
	uint32x4x2_t vld2q_u32 (const uint32_t * __a);
	poly8x16x2_t vld2q_p8 (const poly8_t * __a);
	poly16x8x2_t vld2q_p16 (const poly16_t * __a);
	/*--5、Load a single lane of 2-element structure from memory: vld2 -> 
	loads two elements in a double-vector structure from memory and returns this in 
	the result. The loaded values are from consecutive memory addresses. 
	Elements in the structure that are not loaded are returned in the result unaltered. 
	c is the index of the elements to load.--*/
	int8x8x2_t vld2_lane_s8 (const int8_t * __a, int8x8x2_t __b, const int __c);
	int16x4x2_t vld2_lane_s16 (const int16_t * __a, int16x4x2_t __b, const int __c);
	int32x2x2_t vld2_lane_s32 (const int32_t * __a, int32x2x2_t __b, const int __c);
	float32x2x2_t vld2_lane_f32 (const float32_t * __a, float32x2x2_t __b, const int __c);
	uint8x8x2_t vld2_lane_u8 (const uint8_t * __a, uint8x8x2_t __b, const int __c);
	uint16x4x2_t vld2_lane_u16 (const uint16_t * __a, uint16x4x2_t __b, const int __c);
	uint32x2x2_t vld2_lane_u32 (const uint32_t * __a, uint32x2x2_t __b, const int __c);
	poly8x8x2_t vld2_lane_p8 (const poly8_t * __a, poly8x8x2_t __b, const int __c);
	poly16x4x2_t vld2_lane_p16 (const poly16_t * __a, poly16x4x2_t __b, const int __c);
	int16x8x2_t vld2q_lane_s16 (const int16_t * __a, int16x8x2_t __b, const int __c);
	int32x4x2_t vld2q_lane_s32 (const int32_t * __a, int32x4x2_t __b, const int __c);
	float32x4x2_t vld2q_lane_f32 (const float32_t * __a, float32x4x2_t __b, const int __c);
	uint16x8x2_t vld2q_lane_u16 (const uint16_t * __a, uint16x8x2_t __b, const int __c);
	uint32x4x2_t vld2q_lane_u32 (const uint32_t * __a, uint32x4x2_t __b, const int __c);
	poly16x8x2_t vld2q_lane_p16 (const poly16_t * __a, poly16x8x2_t __b, const int __c);
	/*--6、Load all lanes of 2-element structure with same value from memory: vld2 -> 
	loads 2 elements from memory and returns a double-vector structure. 
	The first element is copied to all lanes of the first vector. 
	The second element is copied to all lanes of the second vector.--*/
	int8x8x2_t vld2_dup_s8 (const int8_t * __a);
	int16x4x2_t vld2_dup_s16 (const int16_t * __a);
	int32x2x2_t vld2_dup_s32 (const int32_t * __a);
	float32x2x2_t vld2_dup_f32 (const float32_t * __a);
	uint8x8x2_t vld2_dup_u8 (const uint8_t * __a);
	uint16x4x2_t vld2_dup_u16 (const uint16_t * __a);
	uint32x2x2_t vld2_dup_u32 (const uint32_t * __a);
	poly8x8x2_t vld2_dup_p8 (const poly8_t * __a);
	poly16x4x2_t vld2_dup_p16 (const poly16_t * __a);
	int64x1x2_t vld2_dup_s64 (const int64_t * __a);
	uint64x1x2_t vld2_dup_u64 (const uint64_t * __a);
	/*--7、Load 3-element structure from memory: vld3 -> 
	loads 3 vectors from memory. 
	It performs a 3-way de-interleave from memory to the vectors.--*/
	int8x8x3_t vld3_s8 (const int8_t * __a);
	int16x4x3_t vld3_s16 (const int16_t * __a);
	int32x2x3_t vld3_s32 (const int32_t * __a);
	float32x2x3_t vld3_f32 (const float32_t * __a);
	uint8x8x3_t vld3_u8 (const uint8_t * __a);
	uint16x4x3_t vld3_u16 (const uint16_t * __a);
	uint32x2x3_t vld3_u32 (const uint32_t * __a);
	poly8x8x3_t vld3_p8 (const poly8_t * __a);
	poly16x4x3_t vld3_p16 (const poly16_t * __a);
	int64x1x3_t vld3_s64 (const int64_t * __a);
	uint64x1x3_t vld3_u64 (const uint64_t * __a);
	int8x16x3_t vld3q_s8 (const int8_t * __a);
	int16x8x3_t vld3q_s16 (const int16_t * __a);
	int32x4x3_t vld3q_s32 (const int32_t * __a);
	float32x4x3_t vld3q_f32 (const float32_t * __a);
	uint8x16x3_t vld3q_u8 (const uint8_t * __a);
	uint16x8x3_t vld3q_u16 (const uint16_t * __a);
	uint32x4x3_t vld3q_u32 (const uint32_t * __a);
	poly8x16x3_t vld3q_p8 (const poly8_t * __a);
	poly16x8x3_t vld3q_p16 (const poly16_t * __a);
	/*--8、Load a single lane of 3-element structure from memory: vld3 -> 
	loads three elements in a triple-vector structure from memory and returns this in the
	result. The loaded values are from consecutive memory addresses. 
	Elements in the structure that are not loaded are returned in the result unaltered.
	c is the index of the element to load.--*/
	int8x8x3_t vld3_lane_s8 (const int8_t * __a, int8x8x3_t __b, const int __c);
	int16x4x3_t vld3_lane_s16 (const int16_t * __a, int16x4x3_t __b, const int __c);
	int32x2x3_t vld3_lane_s32 (const int32_t * __a, int32x2x3_t __b, const int __c);
	float32x2x3_t vld3_lane_f32 (const float32_t * __a, float32x2x3_t __b, const int __c);
	uint8x8x3_t vld3_lane_u8 (const uint8_t * __a, uint8x8x3_t __b, const int __c);
	uint16x4x3_t vld3_lane_u16 (const uint16_t * __a, uint16x4x3_t __b, const int __c);
	uint32x2x3_t vld3_lane_u32 (const uint32_t * __a, uint32x2x3_t __b, const int __c);
	poly8x8x3_t vld3_lane_p8 (const poly8_t * __a, poly8x8x3_t __b, const int __c);
	poly16x4x3_t vld3_lane_p16 (const poly16_t * __a, poly16x4x3_t __b, const int __c);
	int16x8x3_t vld3q_lane_s16 (const int16_t * __a, int16x8x3_t __b, const int __c);
	int32x4x3_t vld3q_lane_s32 (const int32_t * __a, int32x4x3_t __b, const int __c);
	float32x4x3_t vld3q_lane_f32 (const float32_t * __a, float32x4x3_t __b, const int __c);
	uint16x8x3_t vld3q_lane_u16 (const uint16_t * __a, uint16x8x3_t __b, const int __c);
	uint32x4x3_t vld3q_lane_u32 (const uint32_t * __a, uint32x4x3_t __b, const int __c);
	poly16x8x3_t vld3q_lane_p16 (const poly16_t * __a, poly16x8x3_t __b, const int __c);
	/*--9、Load all lanes of 3-element structure with same value from memory: vld3 ->
	loads 3 elements from memory and returns a triple-vector structure. The first element
	is copied to all lanes of the first vector. And similarly the second and third elements 
	are copied to the second and third vectors respectively.--*/
	int8x8x3_t vld3_dup_s8 (const int8_t * __a);
	int16x4x3_t vld3_dup_s16 (const int16_t * __a);
	int32x2x3_t vld3_dup_s32 (const int32_t * __a);
	float32x2x3_t vld3_dup_f32 (const float32_t * __a);
	uint8x8x3_t vld3_dup_u8 (const uint8_t * __a);
	uint16x4x3_t vld3_dup_u16 (const uint16_t * __a);
	uint32x2x3_t vld3_dup_u32 (const uint32_t * __a);
	poly8x8x3_t vld3_dup_p8 (const poly8_t * __a);
	poly16x4x3_t vld3_dup_p16 (const poly16_t * __a);
	int64x1x3_t vld3_dup_s64 (const int64_t * __a);
	uint64x1x3_t vld3_dup_u64 (const uint64_t * __a);
	/*--10、Load 4-element structure from memory: vld4 -> 
	loads 4 vectors from memory. 
	It performs a 4-way de-interleave from memory to the vectors.--*/
	int8x8x4_t vld4_s8 (const int8_t * __a);
	int16x4x4_t vld4_s16 (const int16_t * __a);
	int32x2x4_t vld4_s32 (const int32_t * __a);
	float32x2x4_t vld4_f32 (const float32_t * __a);
	uint8x8x4_t  vld4_u8 (const uint8_t * __a);
	uint16x4x4_t vld4_u16 (const uint16_t * __a);
	uint32x2x4_t vld4_u32 (const uint32_t * __a);
	poly8x8x4_t vld4_p8 (const poly8_t * __a);
	poly16x4x4_t vld4_p16 (const poly16_t * __a);
	int64x1x4_t vld4_s64 (const int64_t * __a);
	uint64x1x4_t vld4_u64 (const uint64_t * __a);
	int8x16x4_t vld4q_s8 (const int8_t * __a);
	int16x8x4_t vld4q_s16 (const int16_t * __a);
	int32x4x4_t vld4q_s32 (const int32_t * __a);
	float32x4x4_t vld4q_f32 (const float32_t * __a);
	uint8x16x4_t vld4q_u8 (const uint8_t * __a);
	uint16x8x4_t vld4q_u16 (const uint16_t * __a);
	uint32x4x4_t vld4q_u32 (const uint32_t * __a);
	poly8x16x4_t vld4q_p8 (const poly8_t * __a);
	poly16x8x4_t vld4q_p16 (const poly16_t * __a);
	/*--11、Load a single lane of 4-element structure from memory: vld4 -> 
	loads four elements in a quad-vector structure from memory and returns this in the result. 
	The loaded values are from consecutive memory addresses.
	Elements in the structure that are not loaded are returned in the result unaltered. 
	c is the index of the element to load.--*/
	int8x8x4_t vld4_lane_s8 (const int8_t * __a, int8x8x4_t __b, const int __c);
	int16x4x4_t vld4_lane_s16 (const int16_t * __a, int16x4x4_t __b, const int __c);
	int32x2x4_t vld4_lane_s32 (const int32_t * __a, int32x2x4_t __b, const int __c);
	float32x2x4_t vld4_lane_f32 (const float32_t * __a, float32x2x4_t __b, const int __c);
	uint8x8x4_t vld4_lane_u8 (const uint8_t * __a, uint8x8x4_t __b, const int __c);
	uint16x4x4_t vld4_lane_u16 (const uint16_t * __a, uint16x4x4_t __b, const int __c);
	uint32x2x4_t vld4_lane_u32 (const uint32_t * __a, uint32x2x4_t __b, const int __c);
	poly8x8x4_t vld4_lane_p8 (const poly8_t * __a, poly8x8x4_t __b, const int __c);
	poly16x4x4_t vld4_lane_p16 (const poly16_t * __a, poly16x4x4_t __b, const int __c);
	int16x8x4_t vld4q_lane_s16 (const int16_t * __a, int16x8x4_t __b, const int __c);
	int32x4x4_t vld4q_lane_s32 (const int32_t * __a, int32x4x4_t __b, const int __c);
	float32x4x4_t vld4q_lane_f32 (const float32_t * __a, float32x4x4_t __b, const int __c);
	uint16x8x4_t vld4q_lane_u16 (const uint16_t * __a, uint16x8x4_t __b, const int __c);
	uint32x4x4_t vld4q_lane_u32 (const uint32_t * __a, uint32x4x4_t __b, const int __c);
	poly16x8x4_t vld4q_lane_p16 (const poly16_t * __a, poly16x8x4_t __b, const int __c);
	/*--12、Load all lanes of 4-element structure with same value from memory: vld4 ->
	loads 4 elements from memory and returns a quad-vector structure. The first element is 
	copied to all lanes of the first vector. And similarly the second, third, and fourth 
	elements are copied to the second, third, and fourth vectors respectively.--*/
	int8x8x4_t vld4_dup_s8 (const int8_t * __a);
	int16x4x4_t vld4_dup_s16 (const int16_t * __a);
	int32x2x4_t vld4_dup_s32 (const int32_t * __a);
	float32x2x4_t vld4_dup_f32 (const float32_t * __a);
	uint8x8x4_t vld4_dup_u8 (const uint8_t * __a);
	uint16x4x4_t vld4_dup_u16 (const uint16_t * __a);
	uint32x2x4_t vld4_dup_u32 (const uint32_t * __a);
	poly8x8x4_t vld4_dup_p8 (const poly8_t * __a);
	poly16x4x4_t vld4_dup_p16 (const poly16_t * __a);
	int64x1x4_t vld4_dup_s64 (const int64_t * __a);
	uint64x1x4_t vld4_dup_u64 (const uint64_t * __a);
	/*****************************************************Store*****************************/
	/*--1、Store a single vector into memory: vst1 -> stores a vector into memory.--*/
	void vst1_s8 (int8_t * __a, int8x8_t __b);
	void vst1_s16 (int16_t * __a, int16x4_t __b);
	void vst1_s32 (int32_t * __a, int32x2_t __b);
	void vst1_s64 (int64_t * __a, int64x1_t __b);
	void vst1_f32 (float32_t * __a, float32x2_t __b);
	void vst1_u8 (uint8_t * __a, uint8x8_t __b);
	void vst1_u16 (uint16_t * __a, uint16x4_t __b);
	void vst1_u32 (uint32_t * __a, uint32x2_t __b);
	void vst1_u64 (uint64_t * __a, uint64x1_t __b);
	void vst1_p8 (poly8_t * __a, poly8x8_t __b);
	void vst1_p16 (poly16_t * __a, poly16x4_t __b);
	void vst1q_s8 (int8_t * __a, int8x16_t __b);
	void vst1q_s16 (int16_t * __a, int16x8_t __b);
	void vst1q_s32 (int32_t * __a, int32x4_t __b);
	void vst1q_s64 (int64_t * __a, int64x2_t __b);
	void vst1q_f32 (float32_t * __a, float32x4_t __b);
	void vst1q_u8 (uint8_t * __a, uint8x16_t __b);
	void vst1q_u16 (uint16_t * __a, uint16x8_t __b);
	void vst1q_u32 (uint32_t * __a, uint32x4_t __b);
	void vst1q_u64 (uint64_t * __a, uint64x2_t __b);
	void vst1q_p8 (poly8_t * __a, poly8x16_t __b);
	void vst1q_p16 (poly16_t * __a, poly16x8_t __b);
	/*--2、Store a single lane into memory: vst1 -> 
	stores one element of the vector into memory. 
	c is the index in the vector to be stored.--*/
	void vst1_lane_s8 (int8_t * __a, int8x8_t __b, const int __c);
	void vst1_lane_s16 (int16_t * __a, int16x4_t __b, const int __c);
	void vst1_lane_s32 (int32_t * __a, int32x2_t __b, const int __c);
	void vst1_lane_f32 (float32_t * __a, float32x2_t __b, const int __c);
	void vst1_lane_u8 (uint8_t * __a, uint8x8_t __b, const int __c);
	void vst1_lane_u16 (uint16_t * __a, uint16x4_t __b, const int __c);
	void vst1_lane_u32 (uint32_t * __a, uint32x2_t __b, const int __c);
	void vst1_lane_p8 (poly8_t * __a, poly8x8_t __b, const int __c);
	void vst1_lane_p16 (poly16_t * __a, poly16x4_t __b, const int __c);
	void vst1_lane_s64 (int64_t * __a, int64x1_t __b, const int __c);
	void vst1_lane_u64 (uint64_t * __a, uint64x1_t __b, const int __c);
	void vst1q_lane_s8 (int8_t * __a, int8x16_t __b, const int __c);
	void vst1q_lane_s16 (int16_t * __a, int16x8_t __b, const int __c);
	void vst1q_lane_s32 (int32_t * __a, int32x4_t __b, const int __c);
	void vst1q_lane_f32 (float32_t * __a, float32x4_t __b, const int __c);
	void vst1q_lane_u8 (uint8_t * __a, uint8x16_t __b, const int __c);
	void vst1q_lane_u16 (uint16_t * __a, uint16x8_t __b, const int __c);
	void vst1q_lane_u32 (uint32_t * __a, uint32x4_t __b, const int __c);
	void vst1q_lane_p8 (poly8_t * __a, poly8x16_t __b, const int __c);
	void vst1q_lane_p16 (poly16_t * __a, poly16x8_t __b, const int __c);
	void vst1q_lane_s64 (int64_t * __a, int64x2_t __b, const int __c);
	void vst1q_lane_u64 (uint64_t * __a, uint64x2_t __b, const int __c);
	/*--3、Store 2 vectors into memory: vst2 -> 
	stores 2 vectors into memory. It interleaves the 2 vectors into memory.--*/
	void vst2_s8 (int8_t * __a, int8x8x2_t __b);
	void vst2_s16 (int16_t * __a, int16x4x2_t __b);
	void vst2_s32 (int32_t * __a, int32x2x2_t __b);
	void vst2_f32 (float32_t * __a, float32x2x2_t __b);
	void vst2_u8 (uint8_t * __a, uint8x8x2_t __b);
	void vst2_u16 (uint16_t * __a, uint16x4x2_t __b);
	void vst2_u32 (uint32_t * __a, uint32x2x2_t __b);
	void vst2_p8 (poly8_t * __a, poly8x8x2_t __b);
	void vst2_p16 (poly16_t * __a, poly16x4x2_t __b);
	void vst2_s64 (int64_t * __a, int64x1x2_t __b);
	void vst2_u64 (uint64_t * __a, uint64x1x2_t __b);
	void vst2q_s8 (int8_t * __a, int8x16x2_t __b);
	void vst2q_s16 (int16_t * __a, int16x8x2_t __b);
	void vst2q_s32 (int32_t * __a, int32x4x2_t __b);
	void vst2q_f32 (float32_t * __a, float32x4x2_t __b);
	void vst2q_u8 (uint8_t * __a, uint8x16x2_t __b);
	void vst2q_u16 (uint16_t * __a, uint16x8x2_t __b);
	void vst2q_u32 (uint32_t * __a, uint32x4x2_t __b);
	void vst2q_p8 (poly8_t * __a, poly8x16x2_t __b);
	void vst2q_p16 (poly16_t * __a, poly16x8x2_t __b);
	/*--4、Store a lane of two elements into memory: vst2 ->
	stores a lane of two elements from a double-vector structure into memory.
	The elements to be stored are from the same lane in the vectors and their index is c.--*/
	void vst2_lane_s8 (int8_t * __a, int8x8x2_t __b, const int __c);
	void vst2_lane_s16 (int16_t * __a, int16x4x2_t __b, const int __c);
	void vst2_lane_s32 (int32_t * __a, int32x2x2_t __b, const int __c);
	void vst2_lane_f32 (float32_t * __a, float32x2x2_t __b, const int __c);
	void vst2_lane_u8 (uint8_t * __a, uint8x8x2_t __b, const int __c);
	void vst2_lane_u16 (uint16_t * __a, uint16x4x2_t __b, const int __c);
	void vst2_lane_u32 (uint32_t * __a, uint32x2x2_t __b, const int __c);
	void vst2_lane_p8 (poly8_t * __a, poly8x8x2_t __b, const int __c);
	void vst2_lane_p16 (poly16_t * __a, poly16x4x2_t __b, const int __c);
	void vst2q_lane_s16 (int16_t * __a, int16x8x2_t __b, const int __c);
	void vst2q_lane_s32 (int32_t * __a, int32x4x2_t __b, const int __c);
	void vst2q_lane_f32 (float32_t * __a, float32x4x2_t __b, const int __c);
	void vst2q_lane_u16 (uint16_t * __a, uint16x8x2_t __b, const int __c);
	void vst2q_lane_u32 (uint32_t * __a, uint32x4x2_t __b, const int __c);
	void vst2q_lane_p16 (poly16_t * __a, poly16x8x2_t __b, const int __c);
	/*--5、Store 3 vectors into memory: vst3 -> 
	stores 3 vectors into memory. It interleaves the 3 vectors into memory.--*/
	void vst3_s8 (int8_t * __a, int8x8x3_t __b);
	void vst3_s16 (int16_t * __a, int16x4x3_t __b);
	void vst3_s32 (int32_t * __a, int32x2x3_t __b);
	void vst3_f32 (float32_t * __a, float32x2x3_t __b);
	void  vst3_u8 (uint8_t * __a, uint8x8x3_t __b);
	void vst3_u16 (uint16_t * __a, uint16x4x3_t __b);
	void vst3_u32 (uint32_t * __a, uint32x2x3_t __b);
	void vst3_p8 (poly8_t * __a, poly8x8x3_t __b);
	void vst3_p16 (poly16_t * __a, poly16x4x3_t __b);
	void vst3_s64 (int64_t * __a, int64x1x3_t __b);
	void vst3_u64 (uint64_t * __a, uint64x1x3_t __b);
	void vst3q_s8 (int8_t * __a, int8x16x3_t __b);
	void vst3q_s16 (int16_t * __a, int16x8x3_t __b);
	void vst3q_s32 (int32_t * __a, int32x4x3_t __b);
	void vst3q_f32 (float32_t * __a, float32x4x3_t __b);
	void vst3q_u8 (uint8_t * __a, uint8x16x3_t __b);
	void vst3q_u16 (uint16_t * __a, uint16x8x3_t __b);
	void vst3q_u32 (uint32_t * __a, uint32x4x3_t __b);
	void vst3q_p8 (poly8_t * __a, poly8x16x3_t __b);
	void vst3q_p16 (poly16_t * __a, poly16x8x3_t __b);
	/*--6、Store a lane of three elements into memory: vst3 ->
	stores a lane of three elements from a triple-vector structure into memory. 
	The elements to be stored are from the same lane in the vectors and their index is c.--*/
	void vst3_lane_s8 (int8_t * __a, int8x8x3_t __b, const int __c);
	void vst3_lane_s16 (int16_t * __a, int16x4x3_t __b, const int __c);
	void vst3_lane_s32 (int32_t * __a, int32x2x3_t __b, const int __c);
	void vst3_lane_f32 (float32_t * __a, float32x2x3_t __b, const int __c);
	void vst3_lane_u8 (uint8_t * __a, uint8x8x3_t __b, const int __c);
	void vst3_lane_u16 (uint16_t * __a, uint16x4x3_t __b, const int __c);
	void vst3_lane_u32 (uint32_t * __a, uint32x2x3_t __b, const int __c);
	void vst3_lane_p8 (poly8_t * __a, poly8x8x3_t __b, const int __c);
	void vst3_lane_p16 (poly16_t * __a, poly16x4x3_t __b, const int __c);
	void vst3q_lane_s16 (int16_t * __a, int16x8x3_t __b, const int __c);
	void vst3q_lane_s32 (int32_t * __a, int32x4x3_t __b, const int __c);
	void vst3q_lane_f32 (float32_t * __a, float32x4x3_t __b, const int __c);
	void vst3q_lane_u16 (uint16_t * __a, uint16x8x3_t __b, const int __c);
	void vst3q_lane_u32 (uint32_t * __a, uint32x4x3_t __b, const int __c);
	void vst3q_lane_p16 (poly16_t * __a, poly16x8x3_t __b, const int __c);
	/*--7、Store 4 vectors into memory: vst4 -> 
	stores 4 vectors into memory. It interleaves the 4 vectors into memory.--*/
	void vst4_s8 (int8_t * __a, int8x8x4_t __b);
	void vst4_s16 (int16_t * __a, int16x4x4_t __b);
	void vst4_s32 (int32_t * __a, int32x2x4_t __b);
	void vst4_f32 (float32_t * __a, float32x2x4_t __b);
	void vst4_u8 (uint8_t * __a, uint8x8x4_t __b);
	void vst4_u16 (uint16_t * __a, uint16x4x4_t __b);
	void vst4_u32 (uint32_t * __a, uint32x2x4_t __b);
	void vst4_p8 (poly8_t * __a, poly8x8x4_t __b);
	void vst4_p16 (poly16_t * __a, poly16x4x4_t __b);
	void vst4_s64 (int64_t * __a, int64x1x4_t __b);
	void vst4_u64 (uint64_t * __a, uint64x1x4_t __b);
	void vst4q_s8 (int8_t * __a, int8x16x4_t __b);
	void vst4q_s16 (int16_t * __a, int16x8x4_t __b);
	void vst4q_s32 (int32_t * __a, int32x4x4_t __b);
	void  vst4q_f32 (float32_t * __a, float32x4x4_t __b);
	void vst4q_u8 (uint8_t * __a, uint8x16x4_t __b);
	void vst4q_u16 (uint16_t * __a, uint16x8x4_t __b);
	void vst4q_u32 (uint32_t * __a, uint32x4x4_t __b);
	void vst4q_p8 (poly8_t * __a, poly8x16x4_t __b);
	void vst4q_p16 (poly16_t * __a, poly16x8x4_t __b);
	/*--8、Store a lane of four elements into memory: vst4 ->
	stores a lane of four elements from a quad-vector structure into memory.
	The elements to be stored are from the same lane in the vectors and their index is c.--*/
	void vst4_lane_s8 (int8_t * __a, int8x8x4_t __b, const int __c);
	void vst4_lane_s16 (int16_t * __a, int16x4x4_t __b, const int __c);
	void vst4_lane_s32 (int32_t * __a, int32x2x4_t __b, const int __c);
	void vst4_lane_f32 (float32_t * __a, float32x2x4_t __b, const int __c);
	void vst4_lane_u8 (uint8_t * __a, uint8x8x4_t __b, const int __c);
	void vst4_lane_u16 (uint16_t * __a, uint16x4x4_t __b, const int __c);
	void vst4_lane_u32 (uint32_t * __a, uint32x2x4_t __b, const int __c);
	void vst4_lane_p8 (poly8_t * __a, poly8x8x4_t __b, const int __c);
	void vst4_lane_p16 (poly16_t * __a, poly16x4x4_t __b, const int __c);
	void vst4q_lane_s16 (int16_t * __a, int16x8x4_t __b, const int __c);
	void vst4q_lane_s32 (int32_t * __a, int32x4x4_t __b, const int __c);
	void vst4q_lane_f32 (float32_t * __a, float32x4x4_t __b, const int __c);
	void vst4q_lane_u16 (uint16_t * __a, uint16x8x4_t __b, const int __c);
	void vst4q_lane_u32 (uint32_t * __a, uint32x4x4_t __b, const int __c);
	void vst4q_lane_p16 (poly16_t * __a, poly16x8x4_t __b, const int __c);
	/*********************************Reinterpret casts(type conversion)********************/
	/*--convert between types: vreinterpret -> treats a vector as having a different 
	datatype, without changing its value.--*/
	poly8x8_t vreinterpret_p8_s8 (int8x8_t __a);
	poly8x8_t vreinterpret_p8_s16 (int16x4_t __a);
	poly8x8_t vreinterpret_p8_s32 (int32x2_t __a);
	poly8x8_t vreinterpret_p8_s64 (int64x1_t __a);
	poly8x8_t vreinterpret_p8_f32 (float32x2_t __a);
	poly8x8_t vreinterpret_p8_u8 (uint8x8_t __a);
	poly8x8_t vreinterpret_p8_u16 (uint16x4_t __a);
	poly8x8_t vreinterpret_p8_u32 (uint32x2_t __a);
	poly8x8_t vreinterpret_p8_u64 (uint64x1_t __a);
	poly8x8_t vreinterpret_p8_p16 (poly16x4_t __a);
	poly8x16_t vreinterpretq_p8_s8 (int8x16_t __a);
	poly8x16_t vreinterpretq_p8_s16 (int16x8_t __a);
	poly8x16_t vreinterpretq_p8_s32 (int32x4_t __a);
	poly8x16_t vreinterpretq_p8_s64 (int64x2_t __a);
	poly8x16_t vreinterpretq_p8_f32 (float32x4_t __a);
	poly8x16_t vreinterpretq_p8_u8 (uint8x16_t __a);
	poly8x16_t vreinterpretq_p8_u16 (uint16x8_t __a);
	poly8x16_t vreinterpretq_p8_u32 (uint32x4_t __a);
	poly8x16_t vreinterpretq_p8_u64 (uint64x2_t __a);
	poly8x16_t vreinterpretq_p8_p16 (poly16x8_t __a);
	poly16x4_t vreinterpret_p16_s8 (int8x8_t __a);
	poly16x4_t vreinterpret_p16_s16 (int16x4_t __a);
	poly16x4_t vreinterpret_p16_s32 (int32x2_t __a);
	poly16x4_t vreinterpret_p16_s64 (int64x1_t __a);
	poly16x4_t vreinterpret_p16_f32 (float32x2_t __a);
	poly16x4_t vreinterpret_p16_u8 (uint8x8_t __a);
	poly16x4_t vreinterpret_p16_u16 (uint16x4_t __a);
	poly16x4_t vreinterpret_p16_u32 (uint32x2_t __a);
	poly16x4_t vreinterpret_p16_u64 (uint64x1_t __a);
	poly16x4_t vreinterpret_p16_p8 (poly8x8_t __a);
	poly16x8_t vreinterpretq_p16_s8 (int8x16_t __a);
	poly16x8_t vreinterpretq_p16_s16 (int16x8_t __a);
	poly16x8_t vreinterpretq_p16_s32 (int32x4_t __a);
	poly16x8_t vreinterpretq_p16_s64 (int64x2_t __a);
	poly16x8_t vreinterpretq_p16_f32 (float32x4_t __a);
	poly16x8_t vreinterpretq_p16_u8 (uint8x16_t __a);
	poly16x8_t vreinterpretq_p16_u16 (uint16x8_t __a);
	poly16x8_t vreinterpretq_p16_u32 (uint32x4_t __a);
	poly16x8_t vreinterpretq_p16_u64 (uint64x2_t __a);
	poly16x8_t vreinterpretq_p16_p8 (poly8x16_t __a);
	float32x2_t vreinterpret_f32_s8 (int8x8_t __a);
	float32x2_t vreinterpret_f32_s16 (int16x4_t __a);
	float32x2_t vreinterpret_f32_s32 (int32x2_t __a);
	float32x2_t vreinterpret_f32_s64 (int64x1_t __a);
	float32x2_t vreinterpret_f32_u8 (uint8x8_t __a);
	float32x2_t vreinterpret_f32_u16 (uint16x4_t __a);
	float32x2_t vreinterpret_f32_u32 (uint32x2_t __a);
	float32x2_t vreinterpret_f32_u64 (uint64x1_t __a);
	float32x2_t vreinterpret_f32_p8 (poly8x8_t __a);
	float32x2_t vreinterpret_f32_p16 (poly16x4_t __a);
	float32x4_t vreinterpretq_f32_s8 (int8x16_t __a);
	float32x4_t vreinterpretq_f32_s16 (int16x8_t __a);
	float32x4_t vreinterpretq_f32_s32 (int32x4_t __a);
	float32x4_t vreinterpretq_f32_s64 (int64x2_t __a);
	float32x4_t vreinterpretq_f32_u8 (uint8x16_t __a);
	float32x4_t vreinterpretq_f32_u16 (uint16x8_t __a);
	float32x4_t vreinterpretq_f32_u32 (uint32x4_t __a);
	float32x4_t vreinterpretq_f32_u64 (uint64x2_t __a);
	float32x4_t vreinterpretq_f32_p8 (poly8x16_t __a);
	float32x4_t vreinterpretq_f32_p16 (poly16x8_t __a);
	int64x1_t vreinterpret_s64_s8 (int8x8_t __a);
	int64x1_t vreinterpret_s64_s16 (int16x4_t __a);
	int64x1_t vreinterpret_s64_s32 (int32x2_t __a);
	int64x1_t vreinterpret_s64_f32 (float32x2_t __a);
	int64x1_t vreinterpret_s64_u8 (uint8x8_t __a);
	int64x1_t vreinterpret_s64_u16 (uint16x4_t __a);
	int64x1_t vreinterpret_s64_u32 (uint32x2_t __a);
	int64x1_t vreinterpret_s64_u64 (uint64x1_t __a);
	int64x1_t vreinterpret_s64_p8 (poly8x8_t __a);
	int64x1_t vreinterpret_s64_p16 (poly16x4_t __a);
	int64x2_t vreinterpretq_s64_s8 (int8x16_t __a);
	int64x2_t vreinterpretq_s64_s16 (int16x8_t __a);
	int64x2_t vreinterpretq_s64_s32 (int32x4_t __a);
	int64x2_t vreinterpretq_s64_f32 (float32x4_t __a);
	int64x2_t vreinterpretq_s64_u8 (uint8x16_t __a);
	int64x2_t vreinterpretq_s64_u16 (uint16x8_t __a);
	int64x2_t vreinterpretq_s64_u32 (uint32x4_t __a);
	int64x2_t vreinterpretq_s64_u64 (uint64x2_t __a);
	int64x2_t vreinterpretq_s64_p8 (poly8x16_t __a);
	int64x2_t vreinterpretq_s64_p16 (poly16x8_t __a);
	uint64x1_t vreinterpret_u64_s8 (int8x8_t __a);
	uint64x1_t vreinterpret_u64_s16 (int16x4_t __a);
	uint64x1_t vreinterpret_u64_s32 (int32x2_t __a);
	uint64x1_t vreinterpret_u64_s64 (int64x1_t __a);
	uint64x1_t vreinterpret_u64_f32 (float32x2_t __a);
	uint64x1_t vreinterpret_u64_u8 (uint8x8_t __a);
	uint64x1_t vreinterpret_u64_u16 (uint16x4_t __a);
	uint64x1_t vreinterpret_u64_u32 (uint32x2_t __a);
	uint64x1_t vreinterpret_u64_p8 (poly8x8_t __a);
	uint64x1_t vreinterpret_u64_p16 (poly16x4_t __a);
	uint64x2_t vreinterpretq_u64_s8 (int8x16_t __a);
	uint64x2_t vreinterpretq_u64_s16 (int16x8_t __a);
	uint64x2_t vreinterpretq_u64_s32 (int32x4_t __a);
	uint64x2_t vreinterpretq_u64_s64 (int64x2_t __a);
	uint64x2_t vreinterpretq_u64_f32 (float32x4_t __a);
	uint64x2_t vreinterpretq_u64_u8 (uint8x16_t __a);
	uint64x2_t vreinterpretq_u64_u16 (uint16x8_t __a);
	uint64x2_t vreinterpretq_u64_u32 (uint32x4_t __a);
	uint64x2_t vreinterpretq_u64_p8 (poly8x16_t __a);
	uint64x2_t vreinterpretq_u64_p16 (poly16x8_t __a);
	int8x8_t vreinterpret_s8_s16 (int16x4_t __a);
	int8x8_t vreinterpret_s8_s32 (int32x2_t __a);
	int8x8_t vreinterpret_s8_s64 (int64x1_t __a);
	int8x8_t vreinterpret_s8_f32 (float32x2_t __a);
	int8x8_t vreinterpret_s8_u8 (uint8x8_t __a);
	int8x8_t vreinterpret_s8_u16 (uint16x4_t __a);
	int8x8_t vreinterpret_s8_u32 (uint32x2_t __a);
	int8x8_t vreinterpret_s8_u64 (uint64x1_t __a);
	int8x8_t vreinterpret_s8_p8 (poly8x8_t __a);
	int8x8_t vreinterpret_s8_p16 (poly16x4_t __a);
	int8x16_t vreinterpretq_s8_s16 (int16x8_t __a);
	int8x16_t vreinterpretq_s8_s32 (int32x4_t __a);
	int8x16_t vreinterpretq_s8_s64 (int64x2_t __a);
	int8x16_t vreinterpretq_s8_f32 (float32x4_t __a);
	int8x16_t vreinterpretq_s8_u8 (uint8x16_t __a);
	int8x16_t vreinterpretq_s8_u16 (uint16x8_t __a);
	int8x16_t vreinterpretq_s8_u32 (uint32x4_t __a);
	int8x16_t vreinterpretq_s8_u64 (uint64x2_t __a);
	int8x16_t vreinterpretq_s8_p8 (poly8x16_t __a);
	int8x16_t vreinterpretq_s8_p16 (poly16x8_t __a);
	int16x4_t vreinterpret_s16_s8 (int8x8_t __a);
	int16x4_t vreinterpret_s16_s32 (int32x2_t __a);
	int16x4_t vreinterpret_s16_s64 (int64x1_t __a);
	int16x4_t vreinterpret_s16_f32 (float32x2_t __a);
	int16x4_t vreinterpret_s16_u8 (uint8x8_t __a);
	int16x4_t vreinterpret_s16_u16 (uint16x4_t __a);
	int16x4_t vreinterpret_s16_u32 (uint32x2_t __a);
	int16x4_t vreinterpret_s16_u64 (uint64x1_t __a);
	int16x4_t vreinterpret_s16_p8 (poly8x8_t __a);
	int16x4_t vreinterpret_s16_p16 (poly16x4_t __a);
	int16x8_t vreinterpretq_s16_s8 (int8x16_t __a);
	int16x8_t vreinterpretq_s16_s32 (int32x4_t __a);
	int16x8_t vreinterpretq_s16_s64 (int64x2_t __a);
	int16x8_t vreinterpretq_s16_f32 (float32x4_t __a);
	int16x8_t vreinterpretq_s16_u8 (uint8x16_t __a);
	int16x8_t vreinterpretq_s16_u16 (uint16x8_t __a);
	int16x8_t vreinterpretq_s16_u32 (uint32x4_t __a);
	int16x8_t vreinterpretq_s16_u64 (uint64x2_t __a);
	int16x8_t vreinterpretq_s16_p8 (poly8x16_t __a);
	int16x8_t vreinterpretq_s16_p16 (poly16x8_t __a);
	int32x2_t vreinterpret_s32_s8 (int8x8_t __a);
	int32x2_t vreinterpret_s32_s16 (int16x4_t __a);
	int32x2_t vreinterpret_s32_s64 (int64x1_t __a);
	int32x2_t vreinterpret_s32_f32 (float32x2_t __a);
	int32x2_t vreinterpret_s32_u8 (uint8x8_t __a);
	int32x2_t vreinterpret_s32_u16 (uint16x4_t __a);
	int32x2_t vreinterpret_s32_u32 (uint32x2_t __a);
	int32x2_t vreinterpret_s32_u64 (uint64x1_t __a);
	int32x2_t vreinterpret_s32_p8 (poly8x8_t __a);
	int32x2_t vreinterpret_s32_p16 (poly16x4_t __a);
	int32x4_t vreinterpretq_s32_s8 (int8x16_t __a);
	int32x4_t vreinterpretq_s32_s16 (int16x8_t __a);
	int32x4_t vreinterpretq_s32_s64 (int64x2_t __a);
	int32x4_t vreinterpretq_s32_f32 (float32x4_t __a);
	int32x4_t vreinterpretq_s32_u8 (uint8x16_t __a);
	int32x4_t vreinterpretq_s32_u16 (uint16x8_t __a);
	int32x4_t vreinterpretq_s32_u32 (uint32x4_t __a);
	int32x4_t vreinterpretq_s32_u64 (uint64x2_t __a);
	int32x4_t vreinterpretq_s32_p8 (poly8x16_t __a);
	int32x4_t vreinterpretq_s32_p16 (poly16x8_t __a);
	uint8x8_t vreinterpret_u8_s8 (int8x8_t __a);
	uint8x8_t vreinterpret_u8_s16 (int16x4_t __a);
	uint8x8_t vreinterpret_u8_s32 (int32x2_t __a);
	uint8x8_t vreinterpret_u8_s64 (int64x1_t __a);
	uint8x8_t vreinterpret_u8_f32 (float32x2_t __a);
	uint8x8_t vreinterpret_u8_u16 (uint16x4_t __a);
	uint8x8_t vreinterpret_u8_u32 (uint32x2_t __a);
	uint8x8_t vreinterpret_u8_u64 (uint64x1_t __a);
	uint8x8_t vreinterpret_u8_p8 (poly8x8_t __a);
	uint8x8_t vreinterpret_u8_p16 (poly16x4_t __a);
	uint8x16_t vreinterpretq_u8_s8 (int8x16_t __a);
	uint8x16_t vreinterpretq_u8_s16 (int16x8_t __a);
	uint8x16_t vreinterpretq_u8_s32 (int32x4_t __a);
	uint8x16_t vreinterpretq_u8_s64 (int64x2_t __a);
	uint8x16_t vreinterpretq_u8_f32 (float32x4_t __a);
	uint8x16_t vreinterpretq_u8_u16 (uint16x8_t __a);
	uint8x16_t vreinterpretq_u8_u32 (uint32x4_t __a);
	uint8x16_t vreinterpretq_u8_u64 (uint64x2_t __a);
	uint8x16_t vreinterpretq_u8_p8 (poly8x16_t __a);
	uint8x16_t vreinterpretq_u8_p16 (poly16x8_t __a);
	uint16x4_t vreinterpret_u16_s8 (int8x8_t __a);
	uint16x4_t vreinterpret_u16_s16 (int16x4_t __a);
	uint16x4_t vreinterpret_u16_s32 (int32x2_t __a);
	uint16x4_t vreinterpret_u16_s64 (int64x1_t __a);
	uint16x4_t vreinterpret_u16_f32 (float32x2_t __a);
	uint16x4_t vreinterpret_u16_u8 (uint8x8_t __a);
	uint16x4_t vreinterpret_u16_u32 (uint32x2_t __a);
	uint16x4_t vreinterpret_u16_u64 (uint64x1_t __a);
	uint16x4_t vreinterpret_u16_p8 (poly8x8_t __a);
	uint16x4_t vreinterpret_u16_p16 (poly16x4_t __a);
	uint16x8_t vreinterpretq_u16_s8 (int8x16_t __a);
	uint16x8_t vreinterpretq_u16_s16 (int16x8_t __a);
	uint16x8_t vreinterpretq_u16_s32 (int32x4_t __a);
	uint16x8_t vreinterpretq_u16_s64 (int64x2_t __a);
	uint16x8_t vreinterpretq_u16_f32 (float32x4_t __a);
	uint16x8_t vreinterpretq_u16_u8 (uint8x16_t __a);
	uint16x8_t vreinterpretq_u16_u32 (uint32x4_t __a);
	uint16x8_t vreinterpretq_u16_u64 (uint64x2_t __a);
	uint16x8_t vreinterpretq_u16_p8 (poly8x16_t __a);
	uint16x8_t vreinterpretq_u16_p16 (poly16x8_t __a);
	uint32x2_t vreinterpret_u32_s8 (int8x8_t __a);
	uint32x2_t vreinterpret_u32_s16 (int16x4_t __a);
	uint32x2_t vreinterpret_u32_s32 (int32x2_t __a);
	uint32x2_t vreinterpret_u32_s64 (int64x1_t __a);
	uint32x2_t vreinterpret_u32_f32 (float32x2_t __a);
	uint32x2_t vreinterpret_u32_u8 (uint8x8_t __a);
	uint32x2_t vreinterpret_u32_u16 (uint16x4_t __a);
	uint32x2_t vreinterpret_u32_u64 (uint64x1_t __a);
	uint32x2_t vreinterpret_u32_p8 (poly8x8_t __a);
	uint32x2_t vreinterpret_u32_p16 (poly16x4_t __a);
	uint32x4_t vreinterpretq_u32_s8 (int8x16_t __a);
	uint32x4_t vreinterpretq_u32_s16 (int16x8_t __a);
	uint32x4_t vreinterpretq_u32_s32 (int32x4_t __a);
	uint32x4_t vreinterpretq_u32_s64 (int64x2_t __a);
	uint32x4_t vreinterpretq_u32_f32 (float32x4_t __a);
	uint32x4_t vreinterpretq_u32_u8 (uint8x16_t __a);
	uint32x4_t vreinterpretq_u32_u16 (uint16x8_t __a);
	uint32x4_t vreinterpretq_u32_u64 (uint64x2_t __a);
	uint32x4_t vreinterpretq_u32_p8 (poly8x16_t __a);
	uint32x4_t vreinterpretq_u32_p16 (poly16x8_t __a);
}
