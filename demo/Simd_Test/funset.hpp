#ifndef FBC_SIMD_TEST_FUNSET_HPP_
#define FBC_SIMD_TEST_FUNSET_HPP_

#include <chrono>

namespace {

class Timer {
public:
	static long long getNowTime() { //  milliseconds
		auto now = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
	}
};

}

// Simd library
int test_bgra2gray();

// SIMD
int test_simd_1();

// OpenMP
int test_openmp_api();
int test_openmp_parallel();
int test_openmp_parallel_for();
int test_openmp_for_schedule();
int test_openmp_parallel_sections();
int test_openmp_parallel_single();
int test_openmp_parallel_critical();
int test_openmp_parallel_atomic();
int test_openmp_parallel_barrier();
int test_openmp_parallel_for_nowait();
int test_openmp_parallel_for_master();
int test_openmp_parallel_for_ordered();
int test_openmp_lock();
int test_openmp_variable_private();
int test_openmp_variable_firstprivate();
int test_openmp_variable_lastprivate();
int test_openmp_variable_flush();
int test_openmp_variable_threadprivate();
int test_openmp_variable_copyin();
int test_openmp_variable_copyprivate();
int test_openmp_variable_reduction();


#endif // FBC_SIMD_TEST_FUNSET_HPP_
