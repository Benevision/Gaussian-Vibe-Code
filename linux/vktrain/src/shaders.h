// SPIR-V of the kernels, compiled from shaders/*.comp at build time (glslangValidator --vn) and embedded.
// Kernels that add floats atomically exist twice: "<name>" (compare-and-swap loop) and "<name>_native"
// (GL_EXT_shader_atomic_float).
#pragma once
#include <cstddef>
#include <cstdint>

namespace vkt {
struct ShaderBlob {
    const char* name;
    const uint32_t* code;
    size_t bytes;
};
const ShaderBlob* findShader(const char* name);
bool hasNativeVariant(const char* name);
}  // namespace vkt
