// Matrix multiplication (C = A * B) done in different ways to compare speed

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matmul.h"

// width of the optimization labels in the output, so their PASS columns line up
#define NAME_WIDTH 26

// matrices
static float *matA;      // input A
static float *matB;      // input B
static float *matRefC;   // result of i-j-k (reference)
static float *matTestC;  // result of the method being tested



//start: helper functions
// allocate memory 
static float *allocateMemory(int rows, int cols)
{
    return (float *)malloc((unsigned int)rows * (unsigned int)cols * sizeof(float));
}

// free memory created by allocateMemory 
static void freeMemory(float *mat)
{
    free(mat);
}

// allocate a matrix and fill it with random values between 0 and 1
static float *createMatrix(int rows, int cols)
{
    float *mat = allocateMemory(rows, cols);
    if (mat == NULL)
    {
        return NULL;
    }
    unsigned int t = (unsigned int)time(NULL);
    for (int i = 0; i < rows * cols; i++)
    {
        unsigned int r = (unsigned int)rand() * t;
        mat[i] = (float)(r % (RAND_MAX + 1u)) / (float)RAND_MAX;  // value between 0 and 1
    }
    return mat;
}

// free all matrices 
static void deleteMatrix(void)
{
    freeMemory(matA);
    freeMemory(matB);
    freeMemory(matRefC);
    freeMemory(matTestC);
}

// set every matrix element value to 0
static void resetMatrix(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        mat[i] = 0.0f;
    }
}

// check testC against refC and print PASS/FAIL with time
static void result(const char *name, int width, const float *refC, const float *testC, double time)
{
    int mismatch = 0;
    for (int i = 0; i < ROWS_C * COLS_C; i++)
    {
        float diff = refC[i] - testC[i];
        float ref = refC[i];
        if (diff < 0)
        {
            diff = -diff;
        }
        if (ref < 0)
        {
            ref = -ref;
        }
        if (diff > (TOLERANCE * ref))
        {
            mismatch++;
        }
    }

    if (mismatch == 0)
    {
        printf("%-*s: PASS : %f (ms)\n", width, name, time * 1000);
    }
    else
    {
        printf("%-*s: FAIL : %f (ms)\n", width, name, time * 1000);
    }
}


// print the matrix sizes and settings used for this run
static void printSetupParams(void)
{
    printf("\nMATRIX MULTIPLICATION: C = A * B\n");
    printf("--------------------------------\n");
    printf("Matrix  A: %d x %d\n", ROWS_A, COLS_A);
    printf("Matrix  B: %d x %d\n", ROWS_B, COLS_B);
    printf("Matrix  C: %d x %d\n", ROWS_C, COLS_C);
    printf("REG  size: %d x %d\n", REG, REG);
    printf("TILE size: %d x %d\n", TILE, TILE);
    printf("Unroll   : %d\n", UNROLL_SIZE);
    printf("Data type: float\n");
    printf("Layout   : row-major\n");
    printf("\n");
}

// check the matrix sizes, print what is wrong
// returns 1 if sizes are valid
static int validateMatrixSizes(void)
{
    int positive = (ROWS_A > 0) && (COLS_A > 0) && (ROWS_B > 0) && (COLS_B > 0);
    int compatible = (COLS_A == ROWS_B);  

    if (!positive)
    {
        printf("Matrix dimensions must be positive\n");
        return 0;
    }
    if (!compatible)
    {
        printf("COLS_A must be equal to ROWS_B\n");
        return 0;
    }
    return 1;
}


// returns 1 if UNROLL_SIZE is 2, 4 or 8
static int validateUnrollSize(void)
{
    int valid = (UNROLL_SIZE == 2) || (UNROLL_SIZE == 4) || (UNROLL_SIZE == 8);

    if (!valid)
    {
        printf("Loop unroll: UNROLL_SIZE must be 2, 4 or 8\n");
        return 0;
    }
    return 1;
}
//end: helper functions

// i-j-k (reference)
// row i of A (vector) . column j of B (vector) = C[i][j] (scalar), dot product
// cache: B is read down a column, every step jumps a full row, so many cache misses (fixed in matMul_transposeB)
// reg  : C[i][j] is written to memory every k step, so extra stores
//        (fix: keep the sum in a local variable, same scalar replacement idea as matMul_scalar)
void matMul_ijk(const float *A, const float *B, float *refC)
{
    clock_t start = clock();

    for (int i = 0; i < ROWS_A; i++)
    {
        for (int j = 0; j < COLS_B; j++)
        {
            refC[i * COLS_B + j] = 0.0f;
            for (int k = 0; k < COLS_A; k++)
            {
                refC[i * COLS_B + j] += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
        }
    }

    double time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("i-j-k: REF  : %f (ms)\n", time * 1000);
}

// i-k-j
// A[i][k] (scalar) * row k of B (vector), added into row i of C (vector)
// cache: no major issue, B and C are read along rows
// reg  : A[i][k] is read again for every j, so extra loads (fixed in matMul_scalar)
//        C[i][j] is loaded and stored again for every k, so extra loads/stores (fixed in matMul_regBlock)
void matMul_ikj(const float *A, const float *B, float *testC, const float *refC)
{
    resetMatrix(testC, ROWS_C, COLS_C);
    clock_t start = clock();

    for (int i = 0; i < ROWS_A; i++)
    {
        for (int k = 0; k < COLS_A; k++)
        {
            for (int j = 0; j < COLS_B; j++)
            {
                testC[i * COLS_B + j] += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
        }
    }

    double time = (double)(clock() - start) / CLOCKS_PER_SEC;
    result("i-k-j", 0, refC, testC, time);
}

// j-i-k (same as i-j-k, outer two loops swapped)
// row i of A (vector) . column j of B (vector) = C[i][j] (scalar), dot product
// cache: B is read down a column, every step jumps a full row, so many cache misses (fixed in matMul_transposeB)
// reg  : C[i][j] is written to memory every k step, so extra stores
//        (fix: keep the sum in a local variable, same scalar replacement idea as matMul_scalar)
void matMul_jik(const float *A, const float *B, float *testC, const float *refC)
{
    resetMatrix(testC, ROWS_C, COLS_C);
    clock_t start = clock();

    for (int j = 0; j < COLS_B; j++)
    {
        for (int i = 0; i < ROWS_A; i++)
        {
            for (int k = 0; k < COLS_A; k++)
            {
                testC[i * COLS_B + j] += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
        }
    }

    double time = (double)(clock() - start) / CLOCKS_PER_SEC;
    result("j-i-k", 0, refC, testC, time);
}

// j-k-i
// B[k][j] (scalar) * column k of A (vector), added into column j of C (vector)
// cache: A and C are both read down a column, every step jumps a full row, 
//          so the most cache misses (fixed in matMul_ikj)
// reg  : B[k][j] is read again for every i, so extra loads
//        (fix: take B[k][j] in a variable, same scalar replacement idea as matMul_scalar)
//        C[i][j] is loaded and stored every step, so extra loads/stores
void matMul_jki(const float *A, const float *B, float *testC, const float *refC)
{
    resetMatrix(testC, ROWS_C, COLS_C);
    clock_t start = clock();

    for (int j = 0; j < COLS_B; j++)
    {
        for (int k = 0; k < COLS_A; k++)
        {
            for (int i = 0; i < ROWS_A; i++)
            {
                testC[i * COLS_B + j] += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
        }
    }

    double time = (double)(clock() - start) / CLOCKS_PER_SEC;
    result("j-k-i", 0, refC, testC, time);
}

// k-i-j 
// A[i][k] (scalar) * row k of B (vector), added into row i of C (vector)
// cache: no major issue, B and C are read along rows
// reg  : A[i][k] is read again for every j, so extra loads (fixed in matMul_scalar)
//        C[i][j] is loaded and stored again for every k, so extra loads/stores (fixed in matMul_regBlock)
void matMul_kij(const float *A, const float *B, float *testC, const float *refC)
{
    resetMatrix(testC, ROWS_C, COLS_C);
    clock_t start = clock();

    for (int k = 0; k < COLS_A; k++)
    {
        for (int i = 0; i < ROWS_A; i++)
        {
            for (int j = 0; j < COLS_B; j++)
            {
                testC[i * COLS_B + j] += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
        }
    }

    double time = (double)(clock() - start) / CLOCKS_PER_SEC;
    result("k-i-j", 0, refC, testC, time);
}

// k-j-i (like j-k-i)
// B[k][j] (scalar) * column k of A (vector), added into column j of C (vector)
// cache: A and C are both read down a column, every step jumps a full row, so the most cache misses (fixed in matMul_ikj)
// reg  : B[k][j] is read again for every i, so extra loads
//        (fix: take B[k][j] in a variable, same scalar replacement idea as matMul_scalar)
//        C[i][j] is loaded and stored every step, so extra loads/stores
void matMul_kji(const float *A, const float *B, float *testC, const float *refC)
{
    resetMatrix(testC, ROWS_C, COLS_C);
    clock_t start = clock();

    for (int k = 0; k < COLS_A; k++)
    {
        for (int j = 0; j < COLS_B; j++)
        {
            for (int i = 0; i < ROWS_A; i++)
            {
                testC[i * COLS_B + j] += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
        }
    }

    double time = (double)(clock() - start) / CLOCKS_PER_SEC;
    result("k-j-i", 0, refC, testC, time);

    printf("\n");
}

// scalar replacement (i-k-j)
// same as i-k-j, but A[i][k] is copied to a local variable once
// so the inner loop reads only B and C from memory, not A
// cache: no major issue, same as i-k-j
// reg  : C[i][j] is still loaded and stored for every k, so extra loads/stores (fixed in matMul_regBlock)
void matMul_scalar(const float *A, const float *B, float *testC, const float *refC)
{
    resetMatrix(testC, ROWS_C, COLS_C);
    clock_t start = clock();

    for (int i = 0; i < ROWS_A; i++)
    {
        for (int k = 0; k < COLS_A; k++)
        {
            float a = A[i * COLS_A + k];
            for (int j = 0; j < COLS_B; j++)
            {
                testC[i * COLS_B + j] += a * B[k * COLS_B + j];
            }
        }
    }

    double time = (double)(clock() - start) / CLOCKS_PER_SEC;
    result("Scalar replacement (i-k-j)", NAME_WIDTH, refC, testC, time);
}

// register blocking (i-j-k)
// work on a small REG x REG block of C at a time, kept in acc (registers)
// for each k, load REG values of A and REG values of B, and use each one REG times
// rows/columns that don't fit in a full block are done at the end the normal way
// cache: each k step jumps to the next row of B, so cache misses on big matrices (fixed in matMul_tileBlock)
// reg  : no major issue, the C block stays in registers and each A and B value is used REG times
// caution: registers are few, if REG is too big they spill to the stack, so extra loads/stores again (register spill)
void matMul_regBlock(const float *A, const float *B, float *testC, const float *refC)
{
    int i, j, k;
    clock_t start = clock();

    for (i = 0; i < ROWS_A - (ROWS_A % REG); i += REG)
    {
        for (j = 0; j < COLS_B - (COLS_B % REG); j += REG)
        {
            float acc[REG][REG] = {0};
            float a[REG];
            float b[REG];

            for (k = 0; k < COLS_A; k++)
            {
                for (int p = 0; p < REG; p++)
                {
                    a[p] = A[(i + p) * COLS_A + k];
                }
                for (int q = 0; q < REG; q++)
                {
                    b[q] = B[k * COLS_B + j + q];
                }

                for (int p = 0; p < REG; p++)
                {
                    for (int q = 0; q < REG; q++)
                    {
                        acc[p][q] += a[p] * b[q];
                    }
                }
            }

            // write the block to C
            for (int p = 0; p < REG; p++)
            {
                for (int q = 0; q < REG; q++)
                {
                    testC[(i + p) * COLS_B + j + q] = acc[p][q];
                }
            }
        }

        // leftover columns for these rows (p = row, q = column)
        for (int p = 0; p < REG; p++)
        {
            for (int q = j; q < COLS_B; q++)
            {
                testC[(i + p) * COLS_B + q] = 0.0f;
                for (k = 0; k < COLS_A; k++)
                {
                    testC[(i + p) * COLS_B + q] += A[(i + p) * COLS_A + k] * B[k * COLS_B + q];
                }
            }
        }
    }

    // leftover rows
    for (; i < ROWS_A; i++)
    {
        for (j = 0; j < COLS_B; j++)
        {
            testC[i * COLS_B + j] = 0.0f;
            for (k = 0; k < COLS_A; k++)
            {
                testC[i * COLS_B + j] += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
        }
    }

    double time = (double)(clock() - start) / CLOCKS_PER_SEC;
    result("Register blocking (i-j-k)", NAME_WIDTH, refC, testC, time);
}

// tiling (i-k-j)
// split the matrices into TILE x TILE tiles, small enough to stay in cache
// i, k, j step tile by tile; p, q, r step inside a tile (p = row, q = k, r = column)
// parts that don't fit in a full tile are done at the end the normal way
// cache: no major issue, each tile is reused while it is still in cache
// reg  : C is loaded and stored for every k, so extra loads/stores (fixed in matMul_regBlock)
// caution: cache is limited, if TILE is too big the tiles don't fit, so cache misses again
void matMul_tileBlock(const float *A, const float *B, float *testC, const float *refC)
{
    int i, j, k;
    resetMatrix(testC, ROWS_C, COLS_C);
    clock_t start = clock();

    for (i = 0; i < ROWS_A - (ROWS_A % TILE); i += TILE)
    {
        for (k = 0; k < COLS_A - (COLS_A % TILE); k += TILE)
        {
            for (j = 0; j < COLS_B - (COLS_B % TILE); j += TILE)
            {
                // one tile of A times one tile of B, added into a tile of C
                for (int p = 0; p < TILE; p++)
                {
                    for (int q = 0; q < TILE; q++)
                    {
                        float a = A[(i + p) * COLS_A + k + q];
                        for (int r = 0; r < TILE; r++)
                        {
                            testC[(i + p) * COLS_B + j + r] += a * B[(k + q) * COLS_B + j + r];
                        }
                    }
                }
            }

            // leftover columns
            for (int p = 0; p < TILE; p++)
            {
                for (int q = 0; q < TILE; q++)
                {
                    for (int r = j; r < COLS_B; r++)
                    {
                        testC[(i + p) * COLS_B + r] += A[(i + p) * COLS_A + k + q] * B[(k + q) * COLS_B + r];
                    }
                }
            }
        }

        // leftover k
        for (int p = 0; p < TILE; p++)
        {
            for (int q = k; q < COLS_A; q++)
            {
                for (j = 0; j < COLS_B; j++)
                {
                    testC[(i + p) * COLS_B + j] += A[(i + p) * COLS_A + q] * B[q * COLS_B + j];
                }
            }
        }
    }

    // leftover rows
    for (; i < ROWS_A; i++)
    {
        for (k = 0; k < COLS_A; k++)
        {
            for (j = 0; j < COLS_B; j++)
            {
                testC[i * COLS_B + j] += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
        }
    }

    double time = (double)(clock() - start) / CLOCKS_PER_SEC;
    result("Tile blocking (i-k-j)", NAME_WIDTH, refC, testC, time);
}

// loop unrolling (i-k-j)
// the j loop does UNROLL_SIZE elements per step instead of 1
// whatever is left at the end of the row is done one by one
// cache: no major issue, same as i-k-j
// reg  : C is still loaded and stored for every k, so extra loads/stores (unrolling only cuts loop checks)
// caution: if UNROLL_SIZE is too big, code gets bigger and needs more registers, so it can get slower (register spill)
void matMul_unroll(const float *A, const float *B, float *testC, const float *refC)
{
    if (validateUnrollSize() == 0)
    {
        return;
    }

    resetMatrix(testC, ROWS_C, COLS_C);
    clock_t start = clock();

    for (int i = 0; i < ROWS_A; i++)
    {
        float *c = &testC[i * COLS_B];  // row i of C

        for (int k = 0; k < COLS_A; k++)
        {
            float a = A[i * COLS_A + k];
            const float *b = &B[k * COLS_B];  // row k of B
            int j = 0;

            for (; j + UNROLL_SIZE <= COLS_B; j += UNROLL_SIZE)
            {
                c[j    ] += a * b[j    ];
                c[j + 1] += a * b[j + 1];
#if UNROLL_SIZE >= 4
                c[j + 2] += a * b[j + 2];
                c[j + 3] += a * b[j + 3];
#endif
#if UNROLL_SIZE == 8
                c[j + 4] += a * b[j + 4];
                c[j + 5] += a * b[j + 5];
                c[j + 6] += a * b[j + 6];
                c[j + 7] += a * b[j + 7];
#endif
            }

            for (; j < COLS_B; j++)
            {
                c[j] += a * b[j];
            }
        }
    }

    double time = (double)(clock() - start) / CLOCKS_PER_SEC;
    result("Loop unroll (i-k-j)", NAME_WIDTH, refC, testC, time);
}

// transpose B first, then i-j-k
// after transpose, column j of B becomes row j of Bt,
// so both A and Bt are read along rows (no column reads)
// the time includes the transpose
// cache: transposing B is one extra pass over B, so extra time and extra memory for Bt
// reg  : C[i][j] is written to memory every k step, so extra stores
//        (fix: keep the sum in a local variable, same scalar replacement idea as matMul_scalar)
void matMul_transposeB(const float *A, const float *B, float *testC, const float *refC)
{
    float *Bt = allocateMemory(ROWS_B, COLS_B);
    if (Bt == NULL)
    {
        printf("Transpose B: memory allocation failed\n");
        return;
    }

    clock_t start = clock();

    for (int k = 0; k < ROWS_B; k++)
    {
        for (int j = 0; j < COLS_B; j++)
        {
            Bt[j * ROWS_B + k] = B[k * COLS_B + j];
        }
    }

    for (int i = 0; i < ROWS_A; i++)
    {
        for (int j = 0; j < COLS_B; j++)
        {
            testC[i * COLS_B + j] = 0.0f;
            for (int k = 0; k < COLS_A; k++)
            {
                testC[i * COLS_B + j] += A[i * COLS_A + k] * Bt[j * ROWS_B + k];
            }
        }
    }

    double time = (double)(clock() - start) / CLOCKS_PER_SEC;
    freeMemory(Bt);
    result("Transpose B (i-j-k)", NAME_WIDTH, refC, testC, time);
}



int main(void)
{
    if (validateMatrixSizes() == 0)
    {
        return 1;
    }

    printSetupParams();

    matA = createMatrix(ROWS_A, COLS_A);
    matB = createMatrix(ROWS_B, COLS_B);
    matRefC = allocateMemory(ROWS_C, COLS_C);
    matTestC = allocateMemory(ROWS_C, COLS_C);
    if (matA == NULL || matB == NULL || matRefC == NULL || matTestC == NULL)
    {
        printf("Memory allocation failed\n");
        deleteMatrix();
        return 1;
    }

    // loop orders
    matMul_ijk(matA, matB, matRefC);
    matMul_ikj(matA, matB, matTestC, matRefC);
    matMul_jik(matA, matB, matTestC, matRefC);
    matMul_jki(matA, matB, matTestC, matRefC);
    matMul_kij(matA, matB, matTestC, matRefC);
    matMul_kji(matA, matB, matTestC, matRefC);

    // optimizations
    matMul_scalar(matA, matB, matTestC, matRefC);
    matMul_regBlock(matA, matB, matTestC, matRefC);
    matMul_tileBlock(matA, matB, matTestC, matRefC);
    matMul_unroll(matA, matB, matTestC, matRefC);
    matMul_transposeB(matA, matB, matTestC, matRefC);

    deleteMatrix();
    return 0;
}
