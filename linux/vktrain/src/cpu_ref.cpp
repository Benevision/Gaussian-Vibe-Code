#include "cpu_ref.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace vkt::ref {

ParamsD toDouble(const Params& p) {
    ParamsD d;
    d.n = p.n;
    for (int k = 0; k < 5; k++) d.g[k].assign(p.group(k).begin(), p.group(k).end());
    return d;
}

Cam makeCam(const Camera& c, const RenderOpts& o) {
    Cam k{};
    for (int r = 0; r < 3; r++) {
        for (int j = 0; j < 3; j++) k.W[r][j] = c.w2c[r][j];
        k.t0[r] = c.w2c[r][3];
    }
    k.fx = c.fx; k.fy = c.fy; k.cx = c.cx; k.cy = c.cy;
    k.width = c.width; k.height = c.height;
    k.tilesX = (c.width + TILE - 1) / TILE; k.tilesY = (c.height + TILE - 1) / TILE;
    // The GPU gets these as float: use the same rounded values so the clamp and dilation agree.
    k.limx = double(float(tanLimitX(c))); k.limy = double(float(tanLimitY(c)));
    k.dil = double(o.dilation);
    k.aa = o.antialiased;
    k.shDeg = o.shDegree;
    for (int j = 0; j < 3; j++) k.bg[j] = o.bg[j];
    c.center(k.campos);
    k.ssimWeight = o.ssimWeight;
    return k;
}

double Proj::q(int k) const { return const_cast<Proj*>(this)->q(k); }
double& Proj::q(int k) {
    switch (k) {
        case 0: return mean[0];
        case 1: return mean[1];
        case 2: return conic[0];
        case 3: return conic[1];
        case 4: return conic[2];
        case 5: return op;
        case 6: return rgb[0];
        case 7: return rgb[1];
        case 8: return rgb[2];
        default: return depth;
    }
}

namespace {

const double C0 = 0.28209479177387814, C1 = 0.4886025119029199;
const double C2[5] = {1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396};
const double C3[7] = {-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
                      1.445305721320277, -0.5900435899266435};

void shBasis(const double d[3], int deg, double b[16]) {
    std::fill(b, b + 16, 0.0);
    b[0] = C0;
    if (deg < 1) return;
    double x = d[0], y = d[1], z = d[2];
    b[1] = -C1 * y; b[2] = C1 * z; b[3] = -C1 * x;
    if (deg < 2) return;
    double xx = x * x, yy = y * y, zz = z * z;
    b[4] = C2[0] * x * y; b[5] = C2[1] * y * z; b[6] = C2[2] * (2 * zz - xx - yy); b[7] = C2[3] * x * z; b[8] = C2[4] * (xx - yy);
    if (deg < 3) return;
    b[9] = C3[0] * y * (3 * xx - yy);
    b[10] = C3[1] * x * y * z;
    b[11] = C3[2] * y * (4 * zz - xx - yy);
    b[12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
    b[13] = C3[4] * x * (4 * zz - xx - yy);
    b[14] = C3[5] * z * (xx - yy);
    b[15] = C3[6] * x * (xx - 3 * yy);
}

}  // namespace

void project(const ParamsD& P, const Cam& cam, int i, Proj& out, bool decide) {
    const double* pos = &P.g[0][size_t(i) * 3];
    const double* ls = &P.g[1][size_t(i) * 3];
    const double* qr = &P.g[2][size_t(i) * 4];
    double op = P.g[3][size_t(i)];
    const double* sh = &P.g[4][size_t(i) * 48];
    if (decide) { out = Proj{}; }
    double t[3];
    for (int r = 0; r < 3; r++) t[r] = cam.W[r][0] * pos[0] + cam.W[r][1] * pos[1] + cam.W[r][2] * pos[2] + cam.t0[r];
    if (decide && t[2] < 0.01) return;
    double s[3] = {std::exp(ls[0]), std::exp(ls[1]), std::exp(ls[2])};
    double ql = std::sqrt(qr[0] * qr[0] + qr[1] * qr[1] + qr[2] * qr[2] + qr[3] * qr[3]);
    double q[4] = {1, 0, 0, 0};
    if (ql > 1e-12) for (int k = 0; k < 4; k++) q[k] = qr[k] / ql;
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
                      {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
                      {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
    double M[3][3], S[3][3];
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) M[r][c] = R[r][c] * s[c];
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) S[r][c] = M[r][0] * M[c][0] + M[r][1] * M[c][1] + M[r][2] * M[c][2];
    double rx = t[0] / t[2], ry = t[1] / t[2];
    out.clamped = rx < -cam.limx || rx > cam.limx || ry < -cam.limy || ry > cam.limy;
    double txz = std::clamp(rx, -cam.limx, cam.limx) * t[2], tyz = std::clamp(ry, -cam.limy, cam.limy) * t[2];
    double iz = 1.0 / t[2];
    double J[2][3] = {{cam.fx * iz, 0, -cam.fx * txz * iz * iz}, {0, cam.fy * iz, -cam.fy * tyz * iz * iz}};
    double T[2][3], TS[2][3];
    for (int r = 0; r < 2; r++) for (int c = 0; c < 3; c++) T[r][c] = J[r][0] * cam.W[0][c] + J[r][1] * cam.W[1][c] + J[r][2] * cam.W[2][c];
    for (int r = 0; r < 2; r++) for (int c = 0; c < 3; c++) TS[r][c] = T[r][0] * S[0][c] + T[r][1] * S[1][c] + T[r][2] * S[2][c];
    double c00 = 0, c01 = 0, c11 = 0;
    for (int k = 0; k < 3; k++) { c00 += TS[0][k] * T[0][k]; c01 += TS[0][k] * T[1][k]; c11 += TS[1][k] * T[1][k]; }
    double det0 = c00 * c11 - c01 * c01;
    double a = c00 + cam.dil, b = c01, c = c11 + cam.dil;
    double det = a * c - b * b;
    if (decide && det <= 1e-12) return;
    double o = 1.0 / (1.0 + std::exp(-op));
    double comp = cam.aa ? std::sqrt(std::max(det0, 0.0) / det) : 1.0;
    double oe = o * comp;
    out.conic[0] = c / det; out.conic[1] = -b / det; out.conic[2] = a / det;
    out.op = oe;
    out.mean[0] = cam.fx * t[0] * iz + cam.cx;
    out.mean[1] = cam.fy * t[1] * iz + cam.cy;
    out.depth = t[2];
    if (decide) {
        double k = std::min(3.0, std::sqrt(std::max(2.0 * std::log(std::max(255.0 * oe, 1.0001)), 0.0)));
        double mid = 0.5 * (a + c);
        double l1 = mid + std::sqrt(std::max(0.1, mid * mid - det));
        double radius = std::ceil(k * std::sqrt(l1));
        if (radius <= 0 || oe < 1.0 / 255.0) return;
        double ex = std::ceil(k * std::sqrt(a)), ey = std::ceil(k * std::sqrt(c));
        int x0 = std::clamp(int(std::floor((out.mean[0] - ex) / TILE)), 0, cam.tilesX);
        int y0 = std::clamp(int(std::floor((out.mean[1] - ey) / TILE)), 0, cam.tilesY);
        int x1 = std::clamp(int(std::floor((out.mean[0] + ex) / TILE)) + 1, 0, cam.tilesX);
        int y1 = std::clamp(int(std::floor((out.mean[1] + ey) / TILE)) + 1, 0, cam.tilesY);
        if (x1 <= x0 || y1 <= y0) return;
        out.rect[0] = x0; out.rect[1] = y0; out.rect[2] = x1; out.rect[3] = y1;
        out.tiles = uint32_t((x1 - x0) * (y1 - y0));
        out.valid = true;
    }
    double dv[3] = {pos[0] - cam.campos[0], pos[1] - cam.campos[1], pos[2] - cam.campos[2]};
    double dl = std::sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);
    for (double& v : dv) v /= dl;
    double bsh[16];
    shBasis(dv, cam.shDeg, bsh);
    int nc = (cam.shDeg + 1) * (cam.shDeg + 1);
    for (int ch = 0; ch < 3; ch++) {
        double v = 0;
        for (int k = 0; k < nc; k++) v += bsh[k] * sh[k * 3 + ch];
        out.rgb[ch] = std::max(v + 0.5, 0.0);
    }
}

std::vector<Proj> projectAll(const ParamsD& p, const Cam& cam, bool decide) {
    std::vector<Proj> out(size_t(p.n));
    for (int i = 0; i < p.n; i++) project(p, cam, i, out[size_t(i)], decide);
    return out;
}

std::vector<int> depthOrder(const std::vector<Proj>& pr) {
    std::vector<int> idx;
    for (size_t i = 0; i < pr.size(); i++)
        if (pr[i].valid) idx.push_back(int(i));
    auto key = [&](int i) { float f = float(pr[size_t(i)].depth); uint32_t u; std::memcpy(&u, &f, 4); return u; };
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return key(a) < key(b); });
    return idx;
}

void rasterDecide(const std::vector<Proj>& pr, const Cam& cam, std::vector<double>& img, std::vector<double>& finalT,
                  std::vector<uint32_t>& nContrib, Structure* st) {
    size_t hw = size_t(cam.width) * cam.height;
    img.assign(4 * hw, 0.0);
    finalT.assign(hw, 1.0);
    nContrib.assign(hw, 0);
    if (st) st->lists.assign(hw, {});
    std::vector<int> order = depthOrder(pr);
    std::vector<std::vector<int>> tiles(size_t(cam.tilesX) * cam.tilesY);
    for (int g : order) {
        const Proj& p = pr[size_t(g)];
        for (int ty = p.rect[1]; ty < p.rect[3]; ty++)
            for (int tx = p.rect[0]; tx < p.rect[2]; tx++) tiles[size_t(ty) * cam.tilesX + tx].push_back(g);
    }
    for (int py = 0; py < cam.height; py++)
        for (int px = 0; px < cam.width; px++) {
            const auto& list = tiles[size_t(py / TILE) * cam.tilesX + px / TILE];
            double pfx = px + 0.5, pfy = py + 0.5;
            double T = 1, C[4] = {0, 0, 0, 0};
            uint32_t contributor = 0, last = 0;
            size_t pid = size_t(py) * cam.width + px;
            for (int g : list) {
                contributor++;
                const Proj& p = pr[size_t(g)];
                double dx = p.mean[0] - pfx, dy = p.mean[1] - pfy;
                double power = -0.5 * (p.conic[0] * dx * dx + p.conic[2] * dy * dy) - p.conic[1] * dx * dy;
                if (power > 0) continue;
                double alpha = std::min(0.99, p.op * std::exp(power));
                if (alpha < 1.0 / 255.0) continue;
                double testT = T * (1 - alpha);
                if (testT < 0.0001) break;
                for (int c = 0; c < 3; c++) C[c] += p.rgb[c] * alpha * T;
                C[3] += p.depth * alpha * T;
                T = testT;
                last = contributor;
                if (st) st->lists[pid].push_back(g);
            }
            for (int c = 0; c < 3; c++) img[c * hw + pid] = C[c] + T * cam.bg[c];
            img[3 * hw + pid] = C[3];
            finalT[pid] = T;
            nContrib[pid] = last;
        }
}

void rasterFrozen(const std::vector<Proj>& pr, const Structure& st, const Cam& cam, std::vector<double>& img) {
    size_t hw = size_t(cam.width) * cam.height;
    img.assign(4 * hw, 0.0);
    for (int py = 0; py < cam.height; py++)
        for (int px = 0; px < cam.width; px++) {
            size_t pid = size_t(py) * cam.width + px;
            double pfx = px + 0.5, pfy = py + 0.5;
            double T = 1, C[4] = {0, 0, 0, 0};
            for (int g : st.lists[pid]) {
                const Proj& p = pr[size_t(g)];
                double dx = p.mean[0] - pfx, dy = p.mean[1] - pfy;
                double power = -0.5 * (p.conic[0] * dx * dx + p.conic[2] * dy * dy) - p.conic[1] * dx * dy;
                double alpha = std::min(0.99, p.op * std::exp(power));
                for (int c = 0; c < 3; c++) C[c] += p.rgb[c] * alpha * T;
                C[3] += p.depth * alpha * T;
                T *= 1 - alpha;
            }
            for (int c = 0; c < 3; c++) img[c * hw + pid] = C[c] + T * cam.bg[c];
            img[3 * hw + pid] = C[3];
        }
}

namespace {
void blur(const std::vector<double>& src, std::vector<double>& dst, int w, int h) {
    static double win[11];
    static bool init = false;
    if (!init) {
        double s = 0;
        for (int k = 0; k < 11; k++) s += win[k] = std::exp(-double((k - 5) * (k - 5)) / (2 * 1.5 * 1.5));
        for (double& v : win) v /= s;
        init = true;
    }
    std::vector<double> tmp(src.size(), 0.0);
    dst.assign(src.size(), 0.0);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double s = 0;
            for (int k = -5; k <= 5; k++) { int xx = x + k; if (xx >= 0 && xx < w) s += win[k + 5] * src[size_t(y) * w + xx]; }
            tmp[size_t(y) * w + x] = s;
        }
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double s = 0;
            for (int k = -5; k <= 5; k++) { int yy = y + k; if (yy >= 0 && yy < h) s += win[k + 5] * tmp[size_t(yy) * w + x]; }
            dst[size_t(y) * w + x] = s;
        }
}
}  // namespace

double loss(const std::vector<double>& img, const std::vector<double>& target, int w, int h, double lam, double* l1out, double* ssimOut) {
    size_t hw = size_t(w) * h;
    const double Cc1 = 0.0001, Cc2 = 0.0009;
    double sumL1 = 0, sumS = 0;
    std::vector<double> X(hw), Y(hw), XX(hw), YY(hw), XY(hw), mx, my, mxx, myy, mxy;
    for (int c = 0; c < 3; c++) {
        for (size_t p = 0; p < hw; p++) {
            double x = img[c * hw + p], y = target[c * hw + p];
            X[p] = x; Y[p] = y; XX[p] = x * x; YY[p] = y * y; XY[p] = x * y;
            sumL1 += std::abs(x - y);
        }
        blur(X, mx, w, h); blur(Y, my, w, h); blur(XX, mxx, w, h); blur(YY, myy, w, h); blur(XY, mxy, w, h);
        for (size_t p = 0; p < hw; p++) {
            double sx = mxx[p] - mx[p] * mx[p], sy = myy[p] - my[p] * my[p], sxy = mxy[p] - mx[p] * my[p];
            sumS += (2 * mx[p] * my[p] + Cc1) * (2 * sxy + Cc2) / ((mx[p] * mx[p] + my[p] * my[p] + Cc1) * (sx + sy + Cc2));
        }
    }
    double numel = 3.0 * hw;
    if (l1out) *l1out = sumL1 / numel;
    if (ssimOut) *ssimOut = sumS / numel;
    return (1 - lam) * sumL1 / numel + lam * (1 - sumS / numel);
}

}  // namespace vkt::ref
