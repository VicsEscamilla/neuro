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

    transpose(rows, cols, C, C);
}


__kernel void add_a_to_b(const int rows, const int cols,
        const __global float* A, __global float* B) {
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C
    const int globalCol = get_global_id(1); // Col ID of C

    if (globalRow < rows && globalCol < cols) {
        B[globalRow*cols + globalCol] += A[globalRow*cols + globalCol];
    }
}


__kernel void product(const int rows, const int cols,
        const __global float* A, const __global float* B,
        __global float* C) {
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C
    const int globalCol = get_global_id(1); // Col ID of C

    if (globalRow < rows && globalCol < cols) {
        C[globalRow*cols + globalCol] = B[globalRow*cols + globalCol] * A[globalRow*cols + globalCol];
    }
}


__kernel void update(const int rows, const int cols, const float rate,
        const __global float* delta, __global float* input) {
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C
    const int globalCol = get_global_id(1); // Col ID of C

    if (globalRow < rows && globalCol < cols) {
        input[globalRow*cols + globalCol] += delta[globalRow*cols + globalCol] * rate;
    }
}


// THIS WONT WORK!! The problem is the global_id trying to match multiple dot products
__kernel void backward(const int input_rows, const int input_cols,
                       const int delta_rows, const int delta_cols,
                       const int weights_rows, const int weights_cols,
                       const int biases_size,
                       const __global float* input,
                       const __global float* delta,
                       const __global float* weights,
                       const __global float* biases,
                       const __global float* d_weights,
                       const __global float* d_biases,
                       __global float* input_trans,
                       __global float* delta_trans,
                       __global float* weights_trans,
                       __global float* output) {

    transpose(input_rows, input_cols, input, input_trans);
    transpose(weights_rows, weights_cols, weights, weights_trans);
    transpose(delta_rows, delta_cols, delta, delta_trans);

    // calculate d_biases
    sum(0, delta_rows, delta_cols, delta, d_biases);
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate d_weights
    // cols and rows from input are backward because we're operating on the transpose
    dot(input_cols, delta_cols, input_rows, input_trans, delta, d_weights);
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate backward output
    // cols and rows from weights are backward because we're operating on the transpose
    dot(delta_rows, weights_rows, delta_cols, delta, weights_trans, output);
}
