// Minimal Vulkan 1.2 compute layer for the SplatWork Vulkan trainer: one device and queue, storage buffers
// (one allocation each, device-local, mapped directly when the memory is host-visible, staged otherwise),
// compute pipelines from the embedded SPIR-V, and synchronous command recording.
#pragma once
#include <vulkan/vulkan.h>

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vkt {

struct Error : std::runtime_error { using std::runtime_error::runtime_error; };
void check(VkResult r, const char* what);

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;   // non-null when the memory is host-visible and coherent
};

struct DeviceInfo {
    std::string name;
    std::string driver;
    uint32_t vendorID = 0, deviceID = 0, apiVersion = 0;
    uint32_t subgroupSize = 0;
    bool subgroupOk = false;          // basic + vote + arithmetic in compute shaders
    bool floatAtomicAdd = false;      // shaderBufferFloat32AtomicAdd (VK_EXT_shader_atomic_float)
    bool hostVisibleDeviceMemory = false;
    bool nativeAtomicsBroken = false;  // advertised, but the startup probe got wrong sums (auto mode fell back to CAS)
    VkDeviceSize maxStorageBufferRange = 0;
};

struct ContextOptions {
    int deviceIndex = -1;             // -1: first discrete GPU, else first device
    enum class Atomics { Auto, Native, Cas } atomics = Atomics::Auto;
};

class Context {
public:
    explicit Context(const ContextOptions& opt = {});
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    DeviceInfo info;
    bool nativeAtomics = false;   // which SPIR-V variant the float-atomic kernels use

    static std::vector<std::string> listDevices();

    Buffer createBuffer(VkDeviceSize bytes);
    void destroy(Buffer& b);
    // Grows b (contents lost) if it is smaller than bytes. Only outside recording.
    void ensure(Buffer& b, VkDeviceSize bytes);
    void upload(Buffer& b, const void* data, size_t bytes, size_t offset = 0);
    void download(const Buffer& b, void* data, size_t bytes, size_t offset = 0);

    // Recording: begin(), then dispatch/fill/copy (each followed by a full compute/transfer barrier),
    // then submitWait(). Descriptor sets are transient (pool reset at begin()).
    void begin();
    void dispatch(const char* kernel, const std::vector<const Buffer*>& bufs, const void* pc, uint32_t pcSize,
                  uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);
    void fill(Buffer& b, uint32_t value, VkDeviceSize bytes = VK_WHOLE_SIZE);
    void copy(const Buffer& src, Buffer& dst, VkDeviceSize bytes);
    void submitWait();
    bool recording() const { return recording_; }

    // Runs shaders/atomic_probe.comp with `count` invocations on the current atomic path; returns the number
    // of wrong sums (0 = correct, both one atomic per subgroup and contended per-lane atomics).
    int probeFloatAtomics(uint32_t count);

private:
    struct Pipeline {
        VkPipeline pipe = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        uint32_t nBindings = 0;
    };
    Pipeline& pipeline(const char* kernel, uint32_t nBindings);
    void barrier();
    uint32_t memoryType(uint32_t typeBits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid = 0) const;
    void transfer(Buffer& dev, bool toDevice, void* host, size_t bytes, size_t offset);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps_{};
    std::map<std::string, Pipeline> pipelines_;
    Buffer staging_;
    bool recording_ = false;
};

}  // namespace vkt
