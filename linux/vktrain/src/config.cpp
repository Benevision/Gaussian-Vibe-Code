#include "config.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <map>

namespace vkt {

using nlohmann::json;

void Region::expanded(double lo[3], double hi[3]) const {
    for (int k = 0; k < 3; k++) {
        double s = max[k] - min[k];
        lo[k] = min[k] - margin * s;
        hi[k] = max[k] + margin * s;
    }
}

namespace {

std::string typeName(const json& v) {
    std::string s = v.dump();
    if (s.size() > 60) s = s.substr(0, 57) + "...";
    return s;
}

[[noreturn]] void bad(const std::string& name, const json& v, const char* expected) {
    throw ConfigError("invalid value for " + name + ": " + typeName(v) + " (expected " + expected + ")");
}

// Same coercion rules as config.py _coerce.
bool toBool(const std::string& name, const json& v) {
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) {
        double d = v.get<double>();
        if (d == 0 || d == 1) return d == 1;
    }
    if (v.is_string()) {
        std::string s = v.get<std::string>();
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (s == "true" || s == "1") return true;
        if (s == "false" || s == "0") return false;
    }
    bad(name, v, "bool");
}

int64_t toInt(const std::string& name, const json& v) {
    if (v.is_number_integer()) return v.get<int64_t>();
    if (v.is_number_float()) {
        double d = v.get<double>();
        if (std::isfinite(d) && d == std::floor(d) && std::abs(d) < 9.2e18) return int64_t(d);
    }
    if (v.is_string()) {   // Python int("12") works too
        const std::string s = v.get<std::string>();
        try {
            size_t at = 0;
            long long r = std::stoll(s, &at);
            while (at < s.size() && std::isspace(static_cast<unsigned char>(s[at]))) at++;
            if (at == s.size()) return r;
        } catch (...) {}
    }
    bad(name, v, "int");
}

double toFloat(const std::string& name, const json& v) {
    double d = NAN;
    if (v.is_number() && !v.is_boolean()) d = v.get<double>();
    else if (v.is_string()) {
        try {
            size_t at = 0;
            d = std::stod(v.get<std::string>(), &at);
        } catch (...) { d = NAN; }
    }
    if (!std::isfinite(d)) bad(name, v, "float");
    return d;
}

std::string toStr(const std::string& name, const json& v) {
    if (!v.is_string()) bad(name, v, "str");
    std::string s = v.get<std::string>();
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

struct Field {
    enum Kind { Int, Float, Bool, Str } kind;
    void* ptr;
};

std::map<std::string, Field> fieldMap(TrainConfig& c) {
    return {
        {"iterations", {Field::Int, &c.iterations}},
        {"maxImageSize", {Field::Int, &c.maxImageSize}},
        {"shDegree", {Field::Int, &c.shDegree}},
        {"shDegreeInterval", {Field::Int, &c.shDegreeInterval}},
        {"strategy", {Field::Str, &c.strategy}},
        {"maxGaussians", {Field::Int, &c.maxGaussians}},
        {"densifyStart", {Field::Int, &c.densifyStart}},
        {"densifyStop", {Field::Int, &c.densifyStop}},
        {"densifyInterval", {Field::Int, &c.densifyInterval}},
        {"densifyGradThreshold", {Field::Float, &c.densifyGradThreshold}},
        {"absGrad", {Field::Bool, &c.absGrad}},
        {"opacityResetInterval", {Field::Int, &c.opacityResetInterval}},
        {"pruneOpacity", {Field::Float, &c.pruneOpacity}},
        {"lrPosition", {Field::Float, &c.lrPosition}},
        {"lrPositionFinal", {Field::Float, &c.lrPositionFinal}},
        {"lrColor", {Field::Float, &c.lrColor}},
        {"lrSHRest", {Field::Float, &c.lrSHRest}},
        {"lrOpacity", {Field::Float, &c.lrOpacity}},
        {"lrScale", {Field::Float, &c.lrScale}},
        {"lrRotation", {Field::Float, &c.lrRotation}},
        {"ssimWeight", {Field::Float, &c.ssimWeight}},
        {"opacityReg", {Field::Float, &c.opacityReg}},
        {"scaleReg", {Field::Float, &c.scaleReg}},
        {"noiseLR", {Field::Float, &c.noiseLR}},
        {"antialiased", {Field::Bool, &c.antialiased}},
        {"depthWeight", {Field::Float, &c.depthWeight}},
        {"background", {Field::Str, &c.background}},
        {"initialOpacity", {Field::Float, &c.initialOpacity}},
        {"randomInitCount", {Field::Int, &c.randomInitCount}},
        {"seed", {Field::Int, &c.seed}},
        {"previewInterval", {Field::Int, &c.previewInterval}},
        {"evalInterval", {Field::Int, &c.evalInterval}},
        {"transientMasking", {Field::Bool, &c.transientMasking}},
        {"transientSensitivity", {Field::Float, &c.transientSensitivity}},
        {"transientStart", {Field::Int, &c.transientStart}},
        {"transientMaxViewGap", {Field::Float, &c.transientMaxViewGap}},
        {"depthEdgeAware", {Field::Bool, &c.depthEdgeAware}},
        {"anisotropyReg", {Field::Float, &c.anisotropyReg}},
        {"anisotropyMaxRatio", {Field::Float, &c.anisotropyMaxRatio}},
        {"normalWeight", {Field::Float, &c.normalWeight}},
        {"normalStart", {Field::Int, &c.normalStart}},
        {"flatnessReg", {Field::Float, &c.flatnessReg}},
        {"depthPriors", {Field::Str, &c.depthPriors}},
        {"monoDepthWeight", {Field::Float, &c.monoDepthWeight}},
        {"priorInitPoints", {Field::Int, &c.priorInitPoints}},
        {"longAxisSplit", {Field::Bool, &c.longAxisSplit}},
        {"poseRefinement", {Field::Bool, &c.poseRefinement}},
        {"poseRefinementLR", {Field::Float, &c.poseRefinementLR}},
        {"poseRefinementReg", {Field::Float, &c.poseRefinementReg}},
        {"poseRefinementStart", {Field::Int, &c.poseRefinementStart}},
        {"appearanceCorrection", {Field::Bool, &c.appearanceCorrection}},
        {"appearanceLR", {Field::Float, &c.appearanceLR}},
        {"appearanceReg", {Field::Float, &c.appearanceReg}},
        {"bilateralGrid", {Field::Bool, &c.bilateralGrid}},
        {"bilateralGridLR", {Field::Float, &c.bilateralGridLR}},
        {"bilateralGridTV", {Field::Float, &c.bilateralGridTV}},
        {"progressiveResolution", {Field::Bool, &c.progressiveResolution}},
        {"decoupledSSIM", {Field::Bool, &c.decoupledSSIM}},
        {"backgroundShell", {Field::Int, &c.backgroundShell}},
        {"optimizerPrecision", {Field::Str, &c.optimizerPrecision}},
        {"skybox", {Field::Str, &c.skybox}},
        {"skyboxResolution", {Field::Int, &c.skyboxResolution}},
        {"skyAlphaWeight", {Field::Float, &c.skyAlphaWeight}},
    };
}

Region parseRegion(const json& v) {
    if (!v.is_object()) throw ConfigError("region must be an object {min, max, margin}");
    Region r;
    auto vec3 = [&](const char* key, double out[3]) {
        auto it = v.find(key);
        if (it == v.end() || !it->is_array() || it->size() != 3)
            throw ConfigError("region.min and region.max must be [x, y, z] number arrays");
        for (int k = 0; k < 3; k++) {
            const json& e = (*it)[size_t(k)];
            if (!e.is_number() || e.is_boolean() || !std::isfinite(e.get<double>()))
                throw ConfigError("region.min and region.max must be [x, y, z] number arrays");
            out[k] = e.get<double>();
        }
    };
    vec3("min", r.min);
    vec3("max", r.max);
    for (int k = 0; k < 3; k++)
        if (r.min[k] >= r.max[k]) throw ConfigError("region.min must be < region.max on every axis");
    auto m = v.find("margin");
    if (m != v.end() && !m->is_null()) {
        if (!m->is_number() || m->is_boolean() || !std::isfinite(m->get<double>()) || m->get<double>() < 0)
            throw ConfigError("region.margin must be a non-negative number");
        r.margin = m->get<double>();
    }
    return r;
}

}  // namespace

TrainConfig TrainConfig::fromJson(const json& j) {
    TrainConfig c;
    if (j.is_null()) return c;
    if (!j.is_object()) throw ConfigError("config must be a JSON object");
    auto fields = fieldMap(c);
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& key = it.key();
        const json& v = it.value();
        if (v.is_null()) continue;   // null = use the default
        if (key == "region") {
            c.region = parseRegion(v);
            c.hasRegion = true;
        } else if (key == "cameraIds") {
            if (!v.is_array()) throw ConfigError("cameraIds must be a list of integers");
            c.cameraIds.clear();
            for (const json& e : v) {
                if (!e.is_number() || e.is_boolean()) throw ConfigError("cameraIds must be a list of integers");
                double d = e.get<double>();
                if (d != std::floor(d)) throw ConfigError("cameraIds must be a list of integers");
                c.cameraIds.push_back(e.is_number_integer() ? e.get<int64_t>() : int64_t(d));
            }
            c.hasCameraIds = true;
        } else {
            auto f = fields.find(key);
            if (f == fields.end()) continue;   // unknown keys are ignored (forward compatibility)
            switch (f->second.kind) {
                case Field::Int: *static_cast<int64_t*>(f->second.ptr) = toInt(key, v); break;
                case Field::Float: *static_cast<double*>(f->second.ptr) = toFloat(key, v); break;
                case Field::Bool: *static_cast<bool*>(f->second.ptr) = toBool(key, v); break;
                case Field::Str: *static_cast<std::string*>(f->second.ptr) = toStr(key, v); break;
            }
        }
        c.explicitKeys.insert(key);
    }
    c.validate();
    return c;
}

void TrainConfig::validate() const {
    auto need = [](bool cond, const std::string& msg) { if (!cond) throw ConfigError(msg); };
    need(iterations >= 1, "iterations must be >= 1");
    need(maxImageSize >= 16, "maxImageSize must be >= 16");
    need(0 <= shDegree && shDegree <= 3, "shDegree must be between 0 and 3");
    need(shDegreeInterval >= 1, "shDegreeInterval must be >= 1");
    need(strategy == "mcmc" || strategy == "adaptive", "strategy must be one of ['mcmc', 'adaptive']");
    need(background == "black" || background == "white" || background == "random",
         "background must be one of ['black', 'white', 'random']");
    need(skybox == "auto" || skybox == "on" || skybox == "off", "skybox must be auto, on or off");
    need(16 <= skyboxResolution && skyboxResolution <= 2048, "skyboxResolution must be 16…2048");
    need(0 <= skyAlphaWeight && skyAlphaWeight <= 10, "skyAlphaWeight must be 0…10");
    need(optimizerPrecision == "fp32" || optimizerPrecision == "compact",
         "optimizerPrecision must be one of ['fp32', 'compact']");
    need(maxGaussians >= 1, "maxGaussians must be >= 1");
    need(densifyInterval >= 1, "densifyInterval must be >= 1");
    need(opacityResetInterval >= 1, "opacityResetInterval must be >= 1");
    need(0.0 < initialOpacity && initialOpacity < 1.0, "initialOpacity must be in (0, 1)");
    need(0.0 <= ssimWeight && ssimWeight <= 1.0, "ssimWeight must be in [0, 1]");
    need(0.0 <= pruneOpacity && pruneOpacity < 1.0, "pruneOpacity must be in [0, 1)");
    need(randomInitCount >= 1, "randomInitCount must be >= 1");
    need(previewInterval >= 0, "previewInterval must be >= 0 (0 disables previews)");
    need(evalInterval >= 0, "evalInterval must be >= 0 (0 disables eval)");
    const std::pair<const char*, double> pos[] = {{"lrPosition", lrPosition}, {"lrPositionFinal", lrPositionFinal},
        {"lrColor", lrColor}, {"lrSHRest", lrSHRest}, {"lrOpacity", lrOpacity}, {"lrScale", lrScale},
        {"lrRotation", lrRotation}, {"bilateralGridLR", bilateralGridLR}};
    for (auto& [name, v] : pos) need(v > 0, std::string(name) + " must be > 0");
    const std::pair<const char*, double> nonneg[] = {{"opacityReg", opacityReg}, {"scaleReg", scaleReg},
        {"noiseLR", noiseLR}, {"depthWeight", depthWeight}, {"densifyGradThreshold", densifyGradThreshold},
        {"bilateralGridTV", bilateralGridTV}, {"backgroundShell", double(backgroundShell)}};
    for (auto& [name, v] : nonneg) need(v >= 0, std::string(name) + " must be >= 0");
}

json TrainConfig::toJson() const {
    TrainConfig copy = *this;
    json j = json::object();
    for (auto& [key, f] : fieldMap(copy)) {
        switch (f.kind) {
            case Field::Int: j[key] = *static_cast<int64_t*>(f.ptr); break;
            case Field::Float: j[key] = *static_cast<double*>(f.ptr); break;
            case Field::Bool: j[key] = *static_cast<bool*>(f.ptr); break;
            case Field::Str: j[key] = *static_cast<std::string*>(f.ptr); break;
        }
    }
    if (hasRegion)
        j["region"] = {{"min", {region.min[0], region.min[1], region.min[2]}},
                       {"max", {region.max[0], region.max[1], region.max[2]}}, {"margin", region.margin}};
    else
        j["region"] = nullptr;
    j["cameraIds"] = hasCameraIds ? json(cameraIds) : json(nullptr);
    return j;
}

int64_t TrainConfig::effectiveDensifyStop() const {
    if (strategy == "mcmc" && !isSet("densifyStop")) return 25000;
    return densifyStop;
}
double TrainConfig::effectiveGradThreshold() const { return densifyGradThreshold * (absGrad ? 4.0 : 1.0); }
double TrainConfig::effectiveOpacityReg() const {
    return strategy == "adaptive" && !isSet("opacityReg") ? 0.0 : opacityReg;
}
double TrainConfig::effectiveScaleReg() const {
    return strategy == "adaptive" && !isSet("scaleReg") ? 0.0 : scaleReg;
}
double TrainConfig::effectiveDepthWeight(bool hasDepth) const {
    if (!hasDepth) return 0.0;
    return isSet("depthWeight") ? depthWeight : 0.1;
}

std::vector<std::string> unsupportedNotes(const TrainConfig& c, const FeatureContext& f) {
    std::vector<std::string> out;
    auto note = [&](const std::string& what) {
        out.push_back(what + ": not supported by the Vulkan trainer yet; training without it");
    };
    if (c.effectiveDepthWeight(f.hasDepth) > 0) note("depthWeight (depth supervision from the photos' depth maps)");
    // trainer.py: the depth-normal loss runs with sensor depth, or with predicted depth when depthPriors allows it
    bool priors = !f.hasDepth && c.normalWeight > 0 && (c.depthPriors == "auto" || c.depthPriors == "on");
    if (c.normalWeight > 0 && (f.hasDepth || priors))
        note("normalWeight (depth-normal consistency loss" + std::string(priors ? ", with predicted depth priors)" : ")"));
    if (priors && c.monoDepthWeight > 0) note("monoDepthWeight (depth loss on predicted depth maps)");
    if (priors && c.priorInitPoints > 0) note("priorInitPoints (starting points from predicted depth)");
    if (c.transientMasking) note("transientMasking (ignoring moving objects)");
    if (c.bilateralGrid) note("bilateralGrid (per-photo local colour correction)");
    else if (c.appearanceCorrection && !f.exposureLocked) note("appearanceCorrection (per-photo colour correction)");
    if (c.decoupledSSIM && (c.bilateralGrid || (c.appearanceCorrection && !f.exposureLocked))) note("decoupledSSIM");
    if (c.poseRefinement) note("poseRefinement (per-photo camera pose refinement)");
    if (c.skybox == "on" || (c.skybox == "auto" && f.hasSky)) note("skybox (learned sky at infinity)");
    if (c.optimizerPrecision == "compact") note("optimizerPrecision=compact (reduced-precision Adam moments; fp32 is used)");
    return out;
}

}  // namespace vkt
