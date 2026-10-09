GEMM - Matrix Multiplication (C = A * B)
========================================

This program multiplies two matrices in different ways and shows how long
each way takes. All ways give the same answer, but the speed is very different.

What it does
------------
- Tries all 6 loop orders (i-j-k, i-k-j, j-i-k, j-k-i, k-i-j, k-j-i).
  The simple i-j-k is the reference, the others are checked against it.
- Scalar replacement: reads A[i][k] once and keeps it in a variable.
- Register blocking: works on a small block of C at a time, kept in registers.
- Tile blocking: splits the matrices into small tiles that fit in the cache.
- Loop unroll and Transpose B: does a few elements per loop step,
  and flips B so it is read row by row.

For each method the program prints the time and PASS or FAIL.

Files
-----
matmul.h       - the settings: matrix sizes, REG (register block size),
                 TILE (tile size), UNROLL_SIZE (2, 4 or 8) and TOLERANCE.
                 Change these if you want to try other values.
matmul.c       - all the code: creating the matrices, every method,
                 the PASS/FAIL check and main().
results.txt    - my results on this machine with different gcc options.
readme.txt     - this file.

How to run
----------
1. If you want, change the settings in matmul.h.
2. Build:  gcc -O2 matmul.c -o matmul
           You can use any optimization option, for example
           -O0, -O1, -O2, -O3 or -O3 -march=native.
3. Run:    matmul.exe   on Windows
           ./matmul     on Linux or macOS

Tested environment
------------------
OS        : Windows 11 Pro 64-bit (also checked on Linux x86 and a Linux ARM64 board)
CPU / RAM : Intel Core Ultra 5 225U, 16 GB
Compilers : gcc 15.2.0, clang, Visual Studio (MSVC)
