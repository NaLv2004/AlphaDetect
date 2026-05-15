void SCAN_function_noadd(float* r, float* a, float* b1, int size);
void SCAN_function(float* r, float* a, float* b1, float* b2, int size);
void SCAN_function2(float* r, float* a, float* b1, float* b2, int size);

void SCAN_R0_FLIP(float** LLR_L, float** LLR_R, int stage, int Count, int node, float* Ma_SC, int* uhat);
void SCAN_R1_FLIP(float** LLR_L, float** LLR_R, int stage, int Count, int node, float* Ma_SC, float& theta, int Flip_bit, int* uhat);

void SCAN_REP_FLIP(float** LLR_L, float** LLR_R, int stage, int Count, int node, float* Ma_SC, float& theta, int Flip_bit, int* uhat);
void SCAN_SPC_FLIP(float** LLR_L, float** LLR_R, int stage, int Count, int node, float* Ma_SC, float& theta, int Flip_bit, int* uhat);

void SCAN_R0_LIST(float** LLR, float** beta_R, int beta_start, int Count, int node, int l, int last_start, float* PM, int count_info);
void SCAN_REP_LIST(float** LLR, float** beta, int node, int& l, int beta_start, int last_start, float* PM, int n, int L, int** p, float* W, int* index, int* better, int* worse, int stage);
void SCAN_R1_LIST(float** LLR, float** beta, int node, int& l, int beta_start, int last_start, float* PM, int n, int L, int** p, float* W, int* index, int* better, int* worse, int* min_index, int stage);
void SCAN_R1_LIST_2(float** LLR, float** beta, int node, int& l, int beta_start, int last_start, float* PM, int n, int L, int** p, float* W, int* index, int* better, int* worse, int* path_number, int stage);
