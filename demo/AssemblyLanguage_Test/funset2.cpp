#include "funset.hpp"
#include "common.hpp"
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <bitset>
#include <string>

int test_asmMain()
{
	asmMain();
	return 0;
}

int test_macro_usage()
{
	macro_usage();
	return 0;
}

int test_cpuid()
{
	// reference: https://github.com/brianrhall/Assembly/tree/master/Appendices/Appendix_G_Using%20CPUID/Program_G.1/x86_64
#ifdef _MSC_VER
	std::bitset<32> features1;		// standard features in EDX
	std::bitset<32> features2;		// standard features in ECX
	std::bitset<32> eFeatures1;		// extended features in EBX
	std::bitset<32> eFeatures2;		// extended features in EDX

	int cpu_info[4];			// for returns in EAX[0], EBX[1], ECX[2], EDX[3]

	__cpuid(cpu_info, 1);			// functionID = 1 (EAX = 1)
	features1 = cpu_info[3];		// standard features1 = EDX
	features2 = cpu_info[2];		// standard features2 = ECX

	__cpuidex(cpu_info, 7, 0);		// functionID = 1 (EAX = 7), subfunctionID = 0 (ECX = 0)
	eFeatures1 = cpu_info[1];		// extended features1 = EBX

	__cpuid(cpu_info, 0x80000001);		// functionID = 80000001h (EAX = 80000001h)
	eFeatures2 = cpu_info[3];		// extended features2 = EDX
#else
	std::bitset<64> features1;     // standard features in RDX
	std::bitset<64> features2;     // standard features in RCX
	std::bitset<64> eFeatures1;    // extended features in RBX
	std::bitset<64> eFeatures2;    // extended features in RDX

	asm("movq $1, %%rax \n\t"                // RAX = 1
		"cpuid \n\t"                         // execute CPUID
		"movq %%rdx, %[features1] \n\t"
		"movq %%rcx, %[features2] \n\t"
		"movq $7, %%rax \n\t"                // RAX = 7
		"xorq %%rcx, %%rcx \n\t"             // RCX = 0
		"cpuid \n\t"
		"movq %%rbx, %[eFeatures1] \n\t"
		"movq $0x80000001, %%rax \n\t"       // RAX = 80000001h
		"cpuid \n\t"
		"movq %%rdx, %[eFeatures2] \n\t"
		:[features1] "=m"(features1),        // outputs
		[features2] "=m"(features2),
		[eFeatures1] "=m"(eFeatures1),
		[eFeatures2] "=m"(eFeatures2)
		:                                    // inputs
		: "rax", "%rbx", "%rcx", "%rdx"          // clobbered registers
		);
#endif

	// binary output of features, output in reverse due to Little-Endian
	fprintf(stdout, "===== CPUID bits (right-to-left) =====\n");
#ifdef _MSC_VER
	std::cout << features1 << " -- EDX bits, EAX=1\n"; // fprintf(stdout, "%s\n", features1.to_string<char, std::string::traits_type, std::string::allocator_type>().c_str());
	std::cout << features2 << " -- ECX bits, EAX=1\n";
	std::cout << eFeatures1 << " -- EBX bits, EAX=7 & ECX=0\n";
	std::cout << eFeatures2 << " -- EDX bits, EAX=80000001h\n\n";
#else
	std::cout << features1 << " -- RDX bits, RAX=1\n";
	std::cout << features2 << " -- RCX bits, RAX=1\n";
	std::cout << eFeatures1 << " -- RBX bits, RAX=7 & RCX=0\n";
	std::cout << eFeatures2 << " -- RDX bits, RAX=80000001h\n\n";
#endif

	auto support = [](int i) {
		if (i == 1) return "Yes";
		else return "No";
	};

	fprintf(stdout, "===== CPU Features =====\n");
	fprintf(stdout, "x87 FPU: %s\n", support(features1[0])); // FPU
	fprintf(stdout, "SYSENTER/SYSEXIT: %s\n", support(features1[11])); // SEP (SYSENTER/SYSEXIT)
	fprintf(stdout, "MMX: %s\n", support(features1[23])); // MMX
	fprintf(stdout, "SSE: %s\n", support(features1[25])); // SSE
	fprintf(stdout, "SSE2: %s\n", support(features1[26])); // SSE2
	fprintf(stdout, "SSE3: %s\n", support(features2[0])); // SSE3
	fprintf(stdout, "SSSE3: %s\n", support(features2[9])); // SSSE3
	fprintf(stdout, "FMA3: %s\n", support(features2[12])); // FMA3
	fprintf(stdout, "SSE4.1: %s\n", support(features2[19])); // SSE4.1
	fprintf(stdout, "SSE4.2: %s\n", support(features2[20])); // SSE4.2
	fprintf(stdout, "AVX: %s\n", support(features2[28])); // AVX
	fprintf(stdout, "F16C: %s\n", support(features2[29])); // F16C (half-precision)
	fprintf(stdout, "RDRAND: %s\n", support(features2[30])); // RDRAND (random number generator)

	fprintf(stdout, "===== Extended Features =====\n");
	fprintf(stdout, "AVX2: %s\n", support(eFeatures1[5])); // AVX2
	fprintf(stdout, "AVX512f: %s\n", support(eFeatures1[16])); // AVX512f
	fprintf(stdout, "AVX512dq: %s\n", support(eFeatures1[17])); // AVX512dq
	fprintf(stdout, "RDSEED: %s\n", support(eFeatures1[18])); // RDSEED
	fprintf(stdout, "AVX512ifma: %s\n", support(eFeatures1[21])); // AVX512ifma

	fprintf(stdout, "===== More Extended Features ======\n");
	fprintf(stdout, "SYSCALL/SYSRET: %s\n", support(eFeatures2[11])); // SYSCALL/SYSRET

	return 0;
}

