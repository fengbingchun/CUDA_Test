#ifndef ASSEMBLY_LANGUAGE_TEST_FUNSET_HPP_
#define ASSEMBLY_LANGUAGE_TEST_FUNSET_HPP_

const double PI = 3.14159265358979323846;

int test_asmMain();
int test_inline_int();
int test_inline_float();
int test_macro_usage();
int test_cpuid(); // 通过CPUID指令来判断处理器是否支持某些功能

#endif // ASSEMBLY_LANGUAGE_TEST_FUNSET_HPP_
