// optimizerPrecision "compact" (trainer.py → compact_adam.py): Adam moments in bf16 for positions, scales,
// rotations, opacities and the SH DC term, and as 8-bit codes with one fp32 scale per splat for the higher SH bands —
// on the GPU (shaders/adam_bf16.comp, adam_sh8.comp), 152 instead of 472 bytes of optimiser state per splat.
//
// The host side of the trainer (densification, the resume file) handles moments as Params rows. In compact mode those
// rows are lossless *carriers* of the GPU words, so row copies and zeroed rows (all the strategies do to moments) mean
// what they mean on the GPU, and a save / load round trip is exact:
//   bf16 groups: the decoded bf16 values (exact in fp32);
//   SH rows: [0..2] the decoded bf16 DC moments; [3..47] the int8 m codes (m1) and uint8 v codes (m2) as floats, with
//            the row's fp32 v scale split into 16-bit halves added as 256 × half to m2[3] and m2[4] (integers < 2²⁴).
#pragma once
#include <cstddef>
#include <cstdint>

namespace vkt::compact {

constexpr int SH_WORDS = 27;   // GPU words per splat of the SH moments

double ratioBound(double b1, double b2);   // compact_adam.py ratio_bound
double vLevel(double code);                // normalised v of a code (0 → 0)

void encodeBf16(const float* m, const float* v, size_t count, uint32_t* words);
void decodeBf16(const uint32_t* words, size_t count, float* m, float* v);
void encodeSHRow(const float* m1, const float* m2, uint32_t* words);    // 48 carriers each → 27 words
void decodeSHRow(const uint32_t* words, float* m1, float* m2);
// The moments a carrier row stands for (tests, diagnostics): 48 values each.
void decodedSHMoments(const uint32_t* words, double bound, double* m, double* v);

// Host reference of the two kernels (the same counter-hash random numbers), for the self-test.
struct StepParams { float lr = 1e-3f, lr2 = 1e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-15f, bc1 = 1, bc2 = 1; uint32_t seed = 0, step = 0; };
void refBf16(float* param, const float* grad, uint32_t* words, size_t count, const StepParams& sp);
void refSHRow(float* param48, const float* grad48, uint32_t* words27, uint32_t row, const StepParams& sp, float bound);

}  // namespace vkt::compact
