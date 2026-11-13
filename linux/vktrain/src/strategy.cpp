#include "strategy.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace vkt {

// ------------------------------------------------------------------ random numbers

std::string HostRng::save() const {
    std::ostringstream ss;
    ss << g;
    return ss.str();
}
void HostRng::load(const std::string& s) {
    std::istringstream ss(s);
    ss >> g;
}

namespace {
inline uint32_t pcgHash(uint32_t v) {
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
inline uint32_t rngKey(uint32_t seed, uint32_t step, uint32_t idx, uint32_t k) {
    return pcgHash(seed ^ pcgHash(step ^ pcgHash(idx * 8u + k)));
}
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double logit(double p) { return std::log(p / (1.0 - p)); }

// Row-major rotation of the normalised quaternion (w, x, y, z), as shaders/common.glsl quat_rot.
void quatRot(const float* qin, double R[3][3]) {
    double l = std::sqrt(double(qin[0]) * qin[0] + double(qin[1]) * qin[1] + double(qin[2]) * qin[2] + double(qin[3]) * qin[3]);
    double w = 1, x = 0, y = 0, z = 0;
    if (l > 1e-12) { w = qin[0] / l; x = qin[1] / l; y = qin[2] / l; z = qin[3] / l; }
    R[0][0] = 1 - 2 * (y * y + z * z); R[0][1] = 2 * (x * y - w * z);     R[0][2] = 2 * (x * z + w * y);
    R[1][0] = 2 * (x * y + w * z);     R[1][1] = 1 - 2 * (x * x + z * z); R[1][2] = 2 * (y * z - w * x);
    R[2][0] = 2 * (x * z - w * y);     R[2][1] = 2 * (y * z + w * x);     R[2][2] = 1 - 2 * (x * x + y * y);
}

void copyRow(const Params& a, int src, Params& b, int dst) {
    for (int k = 0; k < 5; k++) {
        int w = Params::width(k);
        std::memcpy(&b.group(k)[size_t(dst) * w], &a.group(k)[size_t(src) * w], size_t(w) * sizeof(float));
    }
}
void zeroRow(Params& b, int dst) {
    for (int k = 0; k < 5; k++) {
        int w = Params::width(k);
        std::fill_n(&b.group(k)[size_t(dst) * w], w, 0.0f);
    }
}

// Rebuilds s from a list of source rows: params copied from src[j]; moments copied, or zero where zeroMoments[j].
void gatherRows(SplatState& s, const std::vector<int>& src, const std::vector<uint8_t>& zeroMoments) {
    SplatState out;
    int n = int(src.size());
    out.p.resize(n); out.m1.resize(n); out.m2.resize(n);
    for (int j = 0; j < n; j++) {
        copyRow(s.p, src[size_t(j)], out.p, j);
        if (zeroMoments[size_t(j)]) continue;   // resize() zero-filled them
        copyRow(s.m1, src[size_t(j)], out.m1, j);
        copyRow(s.m2, src[size_t(j)], out.m2, j);
    }
    s = std::move(out);
}

// torch.multinomial(weights, n, replacement=True).
std::vector<int> multinomial(const std::vector<double>& w, size_t n, HostRng& rng) {
    std::vector<double> cdf(w.size());
    double acc = 0;
    for (size_t i = 0; i < w.size(); i++) { acc += std::max(w[i], 0.0); cdf[i] = acc; }
    std::vector<int> out(n);
    for (size_t j = 0; j < n; j++) {
        double u = rng.uniform() * acc;
        size_t i = size_t(std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
        out[j] = int(std::min(i, w.size() - 1));
    }
    return out;
}

// Equation (9) of "3D Gaussian Splatting as Markov Chain Monte Carlo" (gsplat relocation_kernel; in double here,
// float there): the opacity and scale factor of each of `ratio` copies that together look like the original.
void relocation(double op, int ratio, double& newOp, double& coeff) {
    static double binoms[51][51];
    static bool init = false;
    if (!init) {
        for (int n = 0; n < 51; n++)
            for (int k = 0; k <= n; k++) binoms[n][k] = (k == 0 || k == n) ? 1.0 : binoms[n - 1][k - 1] + binoms[n - 1][k];
        init = true;
    }
    int nIdx = std::clamp(ratio, 1, 51);
    newOp = 1.0 - std::pow(1.0 - op, 1.0 / nIdx);
    double denom = 0;
    for (int i = 1; i <= nIdx; i++)
        for (int k = 0; k <= i - 1; k++)
            denom += binoms[i - 1][k] * ((k % 2 ? -1.0 : 1.0) / std::sqrt(double(k + 1))) * std::pow(newOp, k + 1);
    coeff = op / denom;
}

// gsplat relocate / sample_add: the sampled rows get the opacity and scale of one of (bincount + 1) copies.
void applyRelocation(SplatState& s, const std::vector<int>& sampled, double minOpacity) {
    std::map<int, int> count;
    for (int i : sampled) count[i]++;
    const double eps = std::numeric_limits<float>::epsilon();
    for (auto& [i, c] : count) {
        double op = sigmoid(s.p.opLogit[size_t(i)]);
        double newOp = 0, coeff = 1;
        relocation(op, c + 1, newOp, coeff);
        newOp = std::clamp(newOp, minOpacity, 1.0 - eps);
        s.p.opLogit[size_t(i)] = float(logit(newOp));
        double lc = std::log(coeff);
        for (int k = 0; k < 3; k++) s.p.logScale[size_t(i) * 3 + k] = float(s.p.logScale[size_t(i) * 3 + k] + lc);
    }
}

// ------------------------------------------------------------------ kd-tree for the kNN scale init

struct KdTree {
    const float* pts;
    std::vector<int> idx;
    struct Node { int lo, hi, axis = -1; float split = 0; int left = -1, right = -1; };
    std::vector<Node> nodes;
    int build(int lo, int hi) {
        Node nd;
        nd.lo = lo; nd.hi = hi;
        int me = int(nodes.size());
        nodes.push_back(nd);
        if (hi - lo <= 12) return me;
        float mn[3] = {INFINITY, INFINITY, INFINITY}, mx[3] = {-INFINITY, -INFINITY, -INFINITY};
        for (int i = lo; i < hi; i++)
            for (int k = 0; k < 3; k++) { mn[k] = std::min(mn[k], pts[idx[size_t(i)] * 3 + k]); mx[k] = std::max(mx[k], pts[idx[size_t(i)] * 3 + k]); }
        int axis = 0;
        for (int k = 1; k < 3; k++) if (mx[k] - mn[k] > mx[axis] - mn[axis]) axis = k;
        if (mx[axis] - mn[axis] <= 0) return me;   // all identical: leaf
        int mid = (lo + hi) / 2;
        std::nth_element(idx.begin() + lo, idx.begin() + mid, idx.begin() + hi,
                         [&](int a, int b) { return pts[a * 3 + axis] < pts[b * 3 + axis]; });
        float split = pts[idx[size_t(mid)] * 3 + axis];
        int l = build(lo, mid), r = build(mid, hi);
        nodes[size_t(me)].axis = axis;
        nodes[size_t(me)].split = split;
        nodes[size_t(me)].left = l;
        nodes[size_t(me)].right = r;
        return me;
    }
    // k smallest squared distances (self included) in best[0..k)
    void query(int node, const float* q, int k, double* best) const {
        const Node& nd = nodes[size_t(node)];
        if (nd.axis < 0) {
            for (int i = nd.lo; i < nd.hi; i++) {
                const float* p = &pts[idx[size_t(i)] * 3];
                double d = 0;
                for (int c = 0; c < 3; c++) d += double(p[c] - q[c]) * double(p[c] - q[c]);
                if (d < best[k - 1]) {
                    int j = k - 1;
                    while (j > 0 && best[j - 1] > d) { best[j] = best[j - 1]; j--; }
                    best[j] = d;
                }
            }
            return;
        }
        double diff = double(q[nd.axis]) - nd.split;
        int first = diff < 0 ? nd.left : nd.right, second = diff < 0 ? nd.right : nd.left;
        query(first, q, k, best);
        if (diff * diff < best[k - 1]) query(second, q, k, best);
    }
};

}  // namespace

float hashNormal(uint32_t seed, uint32_t step, uint32_t idx, uint32_t k) {
    uint32_t a = rngKey(seed, step, idx, 2u * k), b = rngKey(seed, step, idx, 2u * k + 1u);
    float u1 = (float(a >> 8u) + 1.0f) * (1.0f / 16777216.0f);
    float u2 = float(b >> 8u) * (1.0f / 16777216.0f);
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.283185307179586f * u2);
}

std::vector<float> knnMeanDist(const std::vector<float>& xyz, int k) {
    size_t n = xyz.size() / 3;
    if (n <= 1) return std::vector<float>(n, 0.01f);
    k = int(std::min<size_t>(size_t(k), n - 1));
    KdTree t;
    t.pts = xyz.data();
    t.idx.resize(n);
    std::iota(t.idx.begin(), t.idx.end(), 0);
    t.nodes.reserve(2 * n / 6 + 16);
    t.build(0, int(n));
    std::vector<float> out(n);
    std::vector<double> best(size_t(k) + 1);
    for (size_t i = 0; i < n; i++) {
        std::fill(best.begin(), best.end(), INFINITY);
        t.query(0, &xyz[i * 3], k + 1, best.data());
        double s = 0;
        for (int j = 1; j <= k; j++) s += std::sqrt(best[size_t(j)]);   // best[0] is the point itself
        out[i] = float(s / k);
    }
    return out;
}

std::vector<float> backgroundShell(const double centre[3], double r, int64_t n) {
    std::vector<float> out(size_t(n) * 3);
    for (int64_t k = 0; k < n; k++) {
        double y = 1 - 2 * (k + 0.5) / double(n);
        double rad = std::sqrt(std::max(0.0, 1 - y * y));
        double th = M_PI * (3 - std::sqrt(5.0)) * double(k);
        double p[3] = {rad * std::cos(th), y, rad * std::sin(th)};
        for (int c = 0; c < 3; c++) out[size_t(k) * 3 + c] = float(centre[c] + r * p[c]);
    }
    return out;
}

SplatState initSplats(const std::vector<float>& xyz, const std::vector<float>& rgb, float initialOpacity, HostRng& rng) {
    const double SH_C0 = 0.28209479177387814;
    int n = int(xyz.size() / 3);
    SplatState s;
    s.p.resize(n); s.m1.resize(n); s.m2.resize(n);
    std::vector<float> dist = knnMeanDist(xyz);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < 3; k++) s.p.pos[size_t(i) * 3 + k] = xyz[size_t(i) * 3 + k];
        float ls = std::log(std::max(dist[size_t(i)], 1e-7f));
        for (int k = 0; k < 3; k++) s.p.logScale[size_t(i) * 3 + k] = ls;
        for (int k = 0; k < 4; k++) s.p.quat[size_t(i) * 4 + k] = float(rng.uniform());   // torch.rand((n, 4))
        s.p.opLogit[size_t(i)] = float(logit(initialOpacity));
        for (int c = 0; c < 3; c++) s.p.sh[size_t(i) * 48 + c] = float((rgb[size_t(i) * 3 + c] - 0.5) / SH_C0);
    }
    return s;
}

McmcResult mcmcRefine(SplatState& s, const McmcParams& prm, HostRng& rng) {
    McmcResult res;
    int n = s.n();
    // _relocate_gs: dead (opacity ≤ min) splats take the place of copies of live ones sampled by opacity
    std::vector<int> dead, alive;
    std::vector<double> probs;
    for (int i = 0; i < n; i++) {
        double o = sigmoid(s.p.opLogit[size_t(i)]);
        if (o <= prm.minOpacity) dead.push_back(i);
        else { alive.push_back(i); probs.push_back(o); }
    }
    if (!dead.empty() && !alive.empty()) {
        std::vector<int> pick = multinomial(probs, dead.size(), rng);
        std::vector<int> sampled(pick.size());
        for (size_t j = 0; j < pick.size(); j++) sampled[j] = alive[size_t(pick[j])];
        applyRelocation(s, sampled, prm.minOpacity);
        for (size_t j = 0; j < dead.size(); j++) copyRow(s.p, sampled[j], s.p, dead[j]);
        // gsplat zeroes the moments of the sampled rows only; the dead rows keep theirs
        for (int i : sampled) { zeroRow(s.m1, i); zeroRow(s.m2, i); }
        res.relocated = dead.size();
    }
    // _add_new_gs: grow by 5 % up to the cap, new copies of splats sampled by opacity
    int64_t target = std::min<int64_t>(prm.capMax, int64_t(1.05 * n));
    int64_t add = std::max<int64_t>(0, target - n);
    if (add > 0) {
        std::vector<double> all(static_cast<size_t>(n));
        for (int i = 0; i < n; i++) all[size_t(i)] = sigmoid(s.p.opLogit[size_t(i)]);
        std::vector<int> sampled = multinomial(all, size_t(add), rng);
        applyRelocation(s, sampled, prm.minOpacity);
        std::vector<int> src(size_t(n) + sampled.size());
        std::vector<uint8_t> zero(src.size(), 0);
        std::iota(src.begin(), src.begin() + n, 0);
        for (size_t j = 0; j < sampled.size(); j++) { src[size_t(n) + j] = sampled[j]; zero[size_t(n) + j] = 1; }
        gatherRows(s, src, zero);
        res.added = size_t(add);
    }
    return res;
}

void mcmcNoise(SplatState& s, float scaler, uint32_t seed, uint32_t step) {
    for (int i = 0; i < s.n(); i++) {
        float o = 1.0f / (1.0f + std::exp(-s.p.opLogit[size_t(i)]));
        float gate = 1.0f / (1.0f + std::exp(-100.0f * ((1.0f - o) - 0.995f)));
        if (gate * scaler == 0.0f) continue;
        double e[3];
        for (uint32_t k = 0; k < 3; k++) e[k] = double(hashNormal(seed, step, uint32_t(i), k)) * gate * scaler;
        double R[3][3];
        quatRot(&s.p.quat[size_t(i) * 4], R);
        double s2[3];
        for (int k = 0; k < 3; k++) s2[k] = std::exp(2.0 * s.p.logScale[size_t(i) * 3 + k]);
        double u[3];
        for (int c = 0; c < 3; c++) u[c] = (R[0][c] * e[0] + R[1][c] * e[1] + R[2][c] * e[2]) * s2[c];
        for (int r = 0; r < 3; r++) s.p.pos[size_t(i) * 3 + r] += float(R[r][0] * u[0] + R[r][1] * u[1] + R[r][2] * u[2]);
    }
}

size_t spreadCloneGroups(SplatState& s) {
    int n = s.n();
    // groups of bit-identical means (torch.unique(means, dim=0)), ordered like torch: by the mean, then by index
    struct Key {
        float v[3];
        bool operator<(const Key& o) const {
            for (int k = 0; k < 3; k++) if (v[k] != o.v[k]) return v[k] < o.v[k];
            return false;
        }
    };
    std::map<Key, std::vector<int>> groups;
    for (int i = 0; i < n; i++) {
        Key key{{s.p.pos[size_t(i) * 3], s.p.pos[size_t(i) * 3 + 1], s.p.pos[size_t(i) * 3 + 2]}};
        groups[key].push_back(i);
    }
    size_t moved = 0;
    for (auto& [key, members] : groups) {
        if (members.size() < 2) continue;
        double m = double(members.size());
        for (size_t r = 0; r < members.size(); r++) {
            int i = members[r];
            float* ls = &s.p.logScale[size_t(i) * 3];
            int ax = 0;
            for (int k = 1; k < 3; k++) if (ls[k] > ls[ax]) ax = k;
            double sigma = std::exp(double(ls[ax]));
            double R[3][3];
            quatRot(&s.p.quat[size_t(i) * 4], R);
            double t = (double(r) - (m - 1) / 2) * 2 * sigma / m;
            for (int c = 0; c < 3; c++) s.p.pos[size_t(i) * 3 + c] = float(double(key.v[c]) + R[c][ax] * t);   // column ax: that axis in world space
            ls[ax] = float(ls[ax] - 0.5 * std::log(m));
            moved++;
        }
    }
    return moved;
}

AdcResult adcRefine(SplatState& s, const std::vector<float>& grad2d, const std::vector<float>& count, const AdcParams& prm,
                    int64_t step, HostRng& rng) {
    AdcResult res;
    int n = s.n();
    auto maxScale = [&](const Params& p, int i) {
        const float* l = &p.logScale[size_t(i) * 3];
        return std::exp(double(std::max({l[0], l[1], l[2]})));
    };
    // _grow_gs
    std::vector<uint8_t> dupli(size_t(n), 0), split(size_t(n), 0);
    for (int i = 0; i < n; i++) {
        double g = grad2d[size_t(i)] / std::max(count[size_t(i)], 1.0f);
        if (!(g > prm.growGrad2d)) continue;
        bool small = maxScale(s.p, i) <= prm.growScale3d * prm.sceneScale;
        (small ? dupli : split)[size_t(i)] = 1;
    }
    // duplicate: copies appended (fresh moments)
    {
        std::vector<int> src(static_cast<size_t>(n));
        std::iota(src.begin(), src.end(), 0);
        std::vector<uint8_t> zero(size_t(n), 0);
        for (int i = 0; i < n; i++) if (dupli[size_t(i)]) { src.push_back(i); zero.push_back(1); res.duplicated++; }
        if (res.duplicated) gatherRows(s, src, zero);
        split.resize(size_t(s.n()), 0);
    }
    // split: each selected splat becomes two, sampled from its Gaussian, scales / 1.6 (gsplat split,
    // revised_opacity off); the originals are removed and the halves appended after the rest
    {
        std::vector<int> sel, rest;
        for (int i = 0; i < s.n(); i++) (split[size_t(i)] ? sel : rest).push_back(i);
        res.split = sel.size();
        if (!sel.empty()) {
            size_t ns = sel.size();
            std::vector<float> samples(2 * ns * 3);   // torch.randn(2, N, 3) then R · (s ⊙ ε)
            std::vector<double> eps(2 * ns * 3);
            for (double& e : eps) e = rng.normal();
            for (size_t b = 0; b < 2; b++)
                for (size_t j = 0; j < ns; j++) {
                    int i = sel[j];
                    double R[3][3];
                    quatRot(&s.p.quat[size_t(i) * 4], R);
                    double sc[3];
                    for (int k = 0; k < 3; k++) sc[k] = std::exp(double(s.p.logScale[size_t(i) * 3 + k]));
                    for (int r = 0; r < 3; r++) {
                        double v = 0;
                        for (int c = 0; c < 3; c++) v += R[r][c] * sc[c] * eps[(b * ns + j) * 3 + size_t(c)];
                        samples[(b * ns + j) * 3 + size_t(r)] = float(v);
                    }
                }
            std::vector<int> src = rest;
            std::vector<uint8_t> zero(rest.size(), 0);
            for (size_t b = 0; b < 2; b++)
                for (size_t j = 0; j < ns; j++) { src.push_back(sel[j]); zero.push_back(1); }
            Params old = s.p;
            gatherRows(s, src, zero);
            for (size_t b = 0; b < 2; b++)
                for (size_t j = 0; j < ns; j++) {
                    size_t row = rest.size() + b * ns + j;
                    int i = sel[j];
                    for (int k = 0; k < 3; k++) {
                        s.p.pos[row * 3 + size_t(k)] = old.pos[size_t(i) * 3 + size_t(k)] + samples[(b * ns + j) * 3 + size_t(k)];
                        s.p.logScale[row * 3 + size_t(k)] =
                            float(std::log(std::exp(double(old.logScale[size_t(i) * 3 + size_t(k)])) / 1.6));
                    }
                }
        }
    }
    // _prune_gs
    {
        std::vector<int> src;
        std::vector<uint8_t> zero;
        for (int i = 0; i < s.n(); i++) {
            bool prune = sigmoid(s.p.opLogit[size_t(i)]) < prm.pruneOpa;
            if (step > prm.resetEvery) prune = prune || maxScale(s.p, i) > prm.pruneScale3d * prm.sceneScale;
            if (prune) res.pruned++;
            else { src.push_back(i); zero.push_back(0); }
        }
        if (res.pruned) gatherRows(s, src, zero);
    }
    return res;
}

void resetOpacity(SplatState& s, double value) {
    float cap = float(logit(value));
    for (int i = 0; i < s.n(); i++) {
        s.p.opLogit[size_t(i)] = std::min(s.p.opLogit[size_t(i)], cap);
        s.m1.opLogit[size_t(i)] = 0;
        s.m2.opLogit[size_t(i)] = 0;
    }
}

}  // namespace vkt
