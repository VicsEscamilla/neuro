__kernel void forward(const int M, const int N, const int K,
                      const __global float* X,
                      const __global float* W,
                      const __global float* B,
                      __global float* R) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += X[k + globalRow*K] * W[globalCol + N*k];
    }

    // Store the result
    R[globalCol + globalRow*N] = acc + B[globalCol];
}

__kernel void dot(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += A[k + globalRow*K] * B[globalCol + N*k];
    }

    // Store the result
    C[globalCol + globalRow*N] = acc;
}

__kernel void transpose(const int rows, const int cols,
                  const __global float* X,
                  __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C
    const int globalCol = get_global_id(1); // Col ID of C

    if (globalRow < rows && globalCol < cols) {
        C[globalCol*rows + globalRow] = X[globalRow*cols + globalCol];
    }
}

void _transpose(const int rows, const int cols,
                  __global float* X,
                  __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C
    const int globalCol = get_global_id(1); // Col ID of C

    if (globalRow < rows && globalCol < cols) {
        C[globalCol*rows + globalRow] = X[globalRow*cols + globalCol];
    }
}

__kernel void sum(const int dim, const int rows, const int cols,
                  const __global float* X,
                  __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C
    const int globalCol = get_global_id(1); // Col ID of C

    C[globalRow*cols + globalCol] = X[globalRow*cols + globalCol];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = cols%2 ? cols/2 + 1 : cols/2; i > 0; i=i/2) {
        if (globalCol < i && globalCol + i < cols) {
            float a = C[globalRow*cols + globalCol];
            float b = C[globalRow*cols + globalCol + i];
            C[globalRow*cols + globalCol] = a + b;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    _transpose(rows, cols, C, C);
}
