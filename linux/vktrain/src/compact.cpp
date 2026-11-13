#include "compact.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vkt::compact {

namespace {
uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float bitsf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
uint32_t pcg(uint32_t v) {
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
uint32_t key(uint32_t seed, uint32_t step, uint32_t idx, uint32_t k) { return pcg(seed ^ pcg(step ^ pcg(idx * 8u + k))); }
uint32_t toBf16(float x, uint32_t r) { return (fbits(x) + (r & 0xFFFFu)) >> 16; }
float uni(uint32_t h) { return float(h >> 8) * (1.0f / 16777216.0f); }
float vLevelF(float c) { return c > 0.0f ? std::exp2((c - 255.0f) * 0.25f) : 0.0f; }
}  // namespace

double ratioBound(double b1, double b2) {
    double r = b2 > 0 ? b1 * b1 / b2 : INFINITY;
    if (r >= 1) return 16.0;
    return std::sqrt((1 - b1) * (1 - b1) / ((1 - b2) * (1 - r)));
}

double vLevel(double c) { return c > 0 ? std::exp2((c - 255.0) / 4.0) : 0.0; }

void encodeBf16(const float* m, const float* v, size_t count, uint32_t* w) {
    for (size_t i = 0; i < count; i++) w[i] = (fbits(m[i]) >> 16) | (fbits(v[i]) & 0xFFFF0000u);
}

void decodeBf16(const uint32_t* w, size_t count, float* m, float* v) {
    for (size_t i = 0; i < count; i++) { m[i] = bitsf(w[i] << 16); v[i] = bitsf(w[i] & 0xFFFF0000u); }
}

void encodeSHRow(const float* m1, const float* m2, uint32_t* w) {
    encodeBf16(m1, m2, 3, w);
    uint32_t c3 = uint32_t(m2[3]), c4 = uint32_t(m2[4]);
    uint32_t scale = (c3 >> 8) | ((c4 >> 8) << 16);
    std::fill(w + 3, w + 26, 0u);
    for (int j = 0; j < 45; j++) {
        uint32_t cm = uint32_t(int32_t(m1[3 + j])) & 0xFFu;
        uint32_t cv = uint32_t(m2[3 + j]) & 0xFFu;
        w[3 + j / 2] |= (cm | (cv << 8)) << (16 * (j & 1));
    }
    w[26] = scale;
}

void decodeSHRow(const uint32_t* w, float* m1, float* m2) {
    decodeBf16(w, 3, m1, m2);
    for (int j = 0; j < 45; j++) {
        uint32_t pr = (w[3 + j / 2] >> (16 * (j & 1))) & 0xFFFFu;
        m1[3 + j] = float(int8_t(uint8_t(pr & 0xFFu)));
        m2[3 + j] = float(pr >> 8);
    }
    m2[3] += 256.0f * float(w[26] & 0xFFFFu);
    m2[4] += 256.0f * float(w[26] >> 16);
}

void decodedSHMoments(const uint32_t* w, double bound, double* m, double* v) {
    for (int j = 0; j < 3; j++) { m[j] = bitsf(w[j] << 16); v[j] = bitsf(w[j] & 0xFFFF0000u); }
    double s = bitsf(w[26]);
    for (int j = 0; j < 45; j++) {
        uint32_t pr = (w[3 + j / 2] >> (16 * (j & 1))) & 0xFFFFu;
        int cm = int8_t(uint8_t(pr & 0xFFu));
        v[3 + j] = vLevel(double(pr >> 8)) * s;
        m[3 + j] = double(cm) * std::abs(cm) * (bound / (127.0 * 127.0)) * std::sqrt(v[3 + j]);
    }
}

void refBf16(float* param, const float* grad, uint32_t* w, size_t count, const StepParams& sp) {
    for (size_t i = 0; i < count; i++) {
        float g = std::isfinite(grad[i]) ? grad[i] : 0.0f;
        float m = bitsf(w[i] << 16), v = bitsf(w[i] & 0xFFFF0000u);
        m = sp.beta1 * m + (1 - sp.beta1) * g;
        v = sp.beta2 * v + (1 - sp.beta2) * g * g;
        param[i] -= sp.lr * (m / sp.bc1) / (std::sqrt(v / sp.bc2) + sp.eps);
        w[i] = toBf16(m, key(sp.seed, sp.step, uint32_t(i), 0)) | (toBf16(v, key(sp.seed, sp.step, uint32_t(i), 1)) << 16);
    }
}

void refSHRow(float* param, const float* grad, uint32_t* w, uint32_t row, const StepParams& sp, float bound) {
    uint32_t P = row * 48;
    for (uint32_t j = 0; j < 3; j++) {
        float g = std::isfinite(grad[j]) ? grad[j] : 0.0f;
        float m = bitsf(w[j] << 16), v = bitsf(w[j] & 0xFFFF0000u);
        m = sp.beta1 * m + (1 - sp.beta1) * g;
        v = sp.beta2 * v + (1 - sp.beta2) * g * g;
        param[j] -= sp.lr * (m / sp.bc1) / (std::sqrt(v / sp.bc2) + sp.eps);
        w[j] = toBf16(m, key(sp.seed, sp.step, P + j, 0)) | (toBf16(v, key(sp.seed, sp.step, P + j, 1)) << 16);
    }
    float s = bitsf(w[26]);
    float mv[45], vv[45], smax = 0;
    for (uint32_t j = 0; j < 45; j++) {
        uint32_t pr = (w[3 + j / 2] >> (16 * (j & 1))) & 0xFFFFu;
        int cm = int8_t(uint8_t(pr & 0xFFu));
        float v = vLevelF(float(pr >> 8)) * s;
        float m = float(cm) * float(std::abs(cm)) * (bound / (127.0f * 127.0f)) * std::sqrt(v);
        float g = std::isfinite(grad[3 + j]) ? grad[3 + j] : 0.0f;
        m = sp.beta1 * m + (1 - sp.beta1) * g;
        v = sp.beta2 * v + (1 - sp.beta2) * g * g;
        param[3 + j] -= sp.lr2 * (m / sp.bc1) / (std::sqrt(v / sp.bc2) + sp.eps);
        mv[j] = m;
        vv[j] = v;
        smax = std::max(smax, v);
    }
    float safe = smax > 0 ? smax : 1.0f;
    uint32_t words[23] = {};
    for (uint32_t j = 0; j < 45; j++) {
        uint32_t idx = P + 3 + j;
        float a = std::clamp(vv[j] / safe, 0.0f, 1.0f);
        float t = std::log2(std::max(a, 1e-38f)) * 4.0f + 255.0f;
        float lo = std::clamp(std::floor(t), 0.0f, 254.0f);
        float vlo = vLevelF(lo), vhi = vLevelF(lo + 1);
        float p = std::clamp((a - vlo) / (vhi - vlo), 0.0f, 1.0f);
        float cv = lo + (uni(key(sp.seed, sp.step, idx, 2)) < p ? 1.0f : 0.0f);
        float vdec = vLevelF(cv) * smax;
        float u = std::sqrt(vdec);
        float r = u > 0 ? std::clamp(mv[j] / std::max(u * bound, 1e-38f), -1.0f, 1.0f) : 0.0f;
        float am = std::abs(r);
        float mlo = std::clamp(std::floor(std::sqrt(am) * 127.0f), 0.0f, 126.0f);
        float mvlo = (mlo / 127.0f) * (mlo / 127.0f), mvhi = ((mlo + 1) / 127.0f) * ((mlo + 1) / 127.0f);
        float pm = std::clamp((am - mvlo) / (mvhi - mvlo), 0.0f, 1.0f);
        float cmag = mlo + (uni(key(sp.seed, sp.step, idx, 3)) < pm ? 1.0f : 0.0f);
        int cm = r < 0 ? -int(cmag) : (r > 0 ? int(cmag) : 0);
        words[j / 2] |= ((uint32_t(cm) & 0xFFu) | (uint32_t(cv) << 8)) << (16 * (j & 1));
    }
    for (int q = 0; q < 23; q++) w[3 + q] = words[q];
    w[26] = fbits(smax);
}

}  // namespace vkt::compact
