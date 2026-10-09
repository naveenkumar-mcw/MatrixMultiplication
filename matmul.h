#ifndef MATMUL_H
#define MATMUL_H

// matrix dimensions, can be changed
// input Matrix A,B,  Output Matrix C
#define ROWS_A 500
#define COLS_A 500
#define ROWS_B 500
#define COLS_B 500
#define ROWS_C ROWS_A
#define COLS_C COLS_B

// Register blocking size: REG x REG
#define REG 4

// Tile blocking size: TILE x TILE
#define TILE 32

// loop unroll size
#define UNROLL_SIZE 2 // only 2, 4, 8 supported now

// relative tolerance for results comparison
#define TOLERANCE 1e-4f

void matMul_ijk(const float *A, const float *B, float *refC); // i, j, k (reference output for later comparison)
void matMul_ikj(const float *A, const float *B, float *testC, const float *refC); // i, k, j
void matMul_jik(const float *A, const float *B, float *testC, const float *refC); // j, i, k
void matMul_jki(const float *A, const float *B, float *testC, const float *refC); // j, k, i
void matMul_kij(const float *A, const float *B, float *testC, const float *refC); // k, i, j
void matMul_kji(const float *A, const float *B, float *testC, const float *refC); // k, j, i

void matMul_scalar(const float *A, const float *B, float *testC, const float *refC);    // scalar replacement (i-k-j)
void matMul_regBlock(const float *A, const float *B, float *testC, const float *refC);  // register blocking (i-j-k, p q innermost)
void matMul_tileBlock(const float *A, const float *B, float *testC, const float *refC); // tile blocking (tiles i-k-j, inside tile i-k-j)
void matMul_unroll(const float *A, const float *B, float *testC, const float *refC);    // loop unrolling (i-k-j, j unrolled)
void matMul_transposeB(const float *A, const float *B, float *testC, const float *refC); // transpose B, then i-j-k (B read row wise)

#endif // MATMUL_H
