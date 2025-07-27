#include <stddef.h>

// Naive dense matrix multiply. Build with -O3 -march=native for AVX2/NEON
// inspired by Stockfish's NNUE optimisations.
void fc1_forward(const float* restrict weights, const float* restrict bias,
                 const float* restrict input, float* restrict output,
                 size_t out_dim, size_t in_dim) {
    for (size_t i = 0; i < out_dim; ++i) {
        const float* wrow = weights + i * in_dim;
        float sum = bias ? bias[i] : 0.0f;
        for (size_t j = 0; j < in_dim; ++j) {
            sum += wrow[j] * input[j];
        }
        output[i] = sum;
    }
}
