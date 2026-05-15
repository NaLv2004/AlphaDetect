#include <vector>
void BP_function(float* r, float* a, float* b1, float* b2, int size);
void BP_function2(float* r, float* a, float* b1, float* b2, int size);
void add_hard_SIMD(float* a1, float* a2, int* b, int size);
void BP_function_OL1(float* r, float* a, float* b1, float* b2, int size);
void BP_function_OR1(float* r, float* a, float* b1, float* b2, int size);
void BP_function_OL2(float* r, float* a, float* b1, float* b2, int size);
void BP_function_OR2(float* r, float* a, float* b1, float* b2, int size);
void f_function(float* a, int size);
void g_function(float* a, int* b, int size);
void combine(int* a, int* b, int size);
void hard_SIMD(float* a, int* b, int size);
float cal_sum(float* a, int len);
void replace_sum(int* a, int* b, int len);
void replace_LLR(float* a, float* b, int len);
void set_PM(float** a, float* b, int L, int len);
void f_function_index(float* a, float* a_new, int size);
void g_function_index(float* a, int* b, float* a_new, int size);
void combine_index(int* a, int* b, int* a_new, int size);

float min_f(float a, float b);
float abs_f(float a);
float max_f(float a, float b);
int sgn(float a);

