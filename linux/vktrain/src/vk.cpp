#include "vk.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "shaders.h"

namespace vkt {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw Error(std::string(what) + " failed (VkResult " + std::to_string(int(r)) + ")");
}

namespace {

bool hasExtension(const std::vector<VkExtensionProperties>& exts, const char* name) {
    for (auto& e : exts)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

VkInstance makeInstance(bool& portability) {
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data());
    std::vector<const char*> enable;
    portability = hasExtension(exts, "VK_KHR_portability_enumeration");
    if (portability) enable.push_back("VK_KHR_portability_enumeration");

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "splatwork-vk-train";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = uint32_t(enable.size());
    ci.ppEnabledExtensionNames = enable.data();
    if (portability) ci.flags |= 0x00000001;  // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
    VkInstance inst = VK_NULL_HANDLE;
    check(vkCreateInstance(&ci, nullptr, &inst), "vkCreateInstance");
    return inst;
}

std::vector<VkPhysicalDevice> physicalDevices(VkInstance inst) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    return devs;
}

}  // namespace

std::vector<std::string> Context::listDevices() {
    bool port = false;
    VkInstance inst = makeInstance(port);
    std::vector<std::string> out;
    for (auto d : physicalDevices(inst)) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        out.push_back(p.deviceName);
    }
    vkDestroyInstance(inst, nullptr);
    return out;
}

Context::Context(const ContextOptions& opt) {
    bool portability = false;
    instance_ = makeInstance(portability);
    auto devs = physicalDevices(instance_);
    if (devs.empty()) throw Error("no Vulkan device found");
    int pick = opt.deviceIndex;
    if (pick < 0) {
        pick = 0;
        for (size_t i = 0; i < devs.size(); i++) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(devs[i], &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pick = int(i); break; }
        }
    }
    if (pick >= int(devs.size())) throw Error("Vulkan device index out of range");
    phys_ = devs[size_t(pick)];

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> dexts(ne);
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ne, dexts.data());
    bool atomicExt = hasExtension(dexts, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

    VkPhysicalDeviceSubgroupProperties sub{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    sub.pNext = &drv;
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext = &sub;
    vkGetPhysicalDeviceProperties2(phys_, &props);
    info.name = props.properties.deviceName;
    info.driver = std::string(drv.driverName) + " " + drv.driverInfo;
    info.vendorID = props.properties.vendorID;
    info.deviceID = props.properties.deviceID;
    info.apiVersion = props.properties.apiVersion;
    info.subgroupSize = sub.subgroupSize;
    info.maxStorageBufferRange = props.properties.limits.maxStorageBufferRange;
    const VkSubgroupFeatureFlags needOps = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    info.subgroupOk = (sub.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) && (sub.supportedOperations & needOps) == needOps;
    if (VK_API_VERSION_MAJOR(info.apiVersion) == 1 && VK_API_VERSION_MINOR(info.apiVersion) < 2)
        throw Error("Vulkan 1.2 is required (device " + info.name + " reports 1." + std::to_string(VK_API_VERSION_MINOR(info.apiVersion)) + ")");
    if (!info.subgroupOk) throw Error("device " + info.name + " lacks subgroup basic/vote/arithmetic operations in compute shaders");

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    if (atomicExt) {
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &af;
        vkGetPhysicalDeviceFeatures2(phys_, &f2);
        info.floatAtomicAdd = af.shaderBufferFloat32AtomicAdd == VK_TRUE;
    }
    switch (opt.atomics) {
        case ContextOptions::Atomics::Auto: nativeAtomics = info.floatAtomicAdd; break;
        case ContextOptions::Atomics::Cas: nativeAtomics = false; break;
        case ContextOptions::Atomics::Native:
            if (!info.floatAtomicAdd) throw Error("native float atomics requested but the device lacks shaderBufferFloat32AtomicAdd");
            nativeAtomics = true;
            break;
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &nq, qf.data());
    bool found = false;
    for (uint32_t i = 0; i < nq && !found; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamily_ = i; found = true; }
    if (!found) throw Error("no compute queue");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queueFamily_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    std::vector<const char*> enable;
    if (hasExtension(dexts, "VK_KHR_portability_subset")) enable.push_back("VK_KHR_portability_subset");
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT afEnable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    if (nativeAtomics) {
        enable.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        afEnable.shaderBufferFloat32AtomicAdd = VK_TRUE;
        dci.pNext = &afEnable;
    }
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(enable.size());
    dci.ppEnabledExtensionNames = enable.data();
    check(vkCreateDevice(phys_, &dci, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);
    vkGetPhysicalDeviceMemoryProperties(phys_, &memProps_);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = queueFamily_;
    check(vkCreateCommandPool(device_, &cpi, nullptr, &cmdPool_), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cmdPool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device_, &cai, &cmd_), "vkAllocateCommandBuffers");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(device_, &fci, nullptr, &fence_), "vkCreateFence");

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 65536};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 4096;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    check(vkCreateDescriptorPool(device_, &dpi, nullptr, &descPool_), "vkCreateDescriptorPool");

    uint32_t dt = memoryType(~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkMemoryPropertyFlags f = memProps_.memoryTypes[dt].propertyFlags;
    info.hostVisibleDeviceMemory = (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Trust but verify: some drivers advertise float atomic add and lose updates under contention (seen on
    // Mesa lavapipe 25.0 / aarch64). In auto mode a failed probe falls back to the compare-and-swap kernels.
    if (nativeAtomics && opt.atomics == ContextOptions::Atomics::Auto && probeFloatAtomics(100003) != 0) {
        nativeAtomics = false;
        info.nativeAtomicsBroken = true;
    }
}

Context::~Context() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        destroy(staging_);
        for (auto& [k, p] : pipelines_) {
            vkDestroyPipeline(device_, p.pipe, nullptr);
            vkDestroyPipelineLayout(device_, p.layout, nullptr);
            vkDestroyDescriptorSetLayout(device_, p.dsl, nullptr);
        }
        vkDestroyDescriptorPool(device_, descPool_, nullptr);
        vkDestroyFence(device_, fence_, nullptr);
        vkDestroyCommandPool(device_, cmdPool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

uint32_t Context::memoryType(uint32_t typeBits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) const {
    for (int pass = 0; pass < 2; pass++)
        for (uint32_t i = 0; i < memProps_.memoryTypeCount; i++) {
            if (!(typeBits & (1u << i))) continue;
            VkMemoryPropertyFlags f = memProps_.memoryTypes[i].propertyFlags;
            if ((f & want) != want) continue;
            if (pass == 0 && (f & avoid)) continue;
            return i;
        }
    throw Error("no suitable Vulkan memory type");
}

Buffer Context::createBuffer(VkDeviceSize bytes) {
    Buffer b;
    b.size = std::max<VkDeviceSize>(bytes, 16);
    b.size = (b.size + 15) & ~VkDeviceSize(15);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = b.size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device_, &bci, nullptr, &b.buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, b.buf, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkResult r = vkAllocateMemory(device_, &mai, nullptr, &b.mem);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(device_, b.buf, nullptr);
        throw Error("out of GPU memory allocating " + std::to_string(bytes) + " bytes");
    }
    check(vkBindBufferMemory(device_, b.buf, b.mem, 0), "vkBindBufferMemory");
    VkMemoryPropertyFlags f = memProps_.memoryTypes[mai.memoryTypeIndex].propertyFlags;
    if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        check(vkMapMemory(device_, b.mem, 0, VK_WHOLE_SIZE, 0, &b.mapped), "vkMapMemory");
    return b;
}

void Context::destroy(Buffer& b) {
    if (b.buf) vkDestroyBuffer(device_, b.buf, nullptr);
    if (b.mem) vkFreeMemory(device_, b.mem, nullptr);
    b = Buffer{};
}

void Context::ensure(Buffer& b, VkDeviceSize bytes) {
    if (recording_) throw Error("ensure() while recording");
    if (b.buf && b.size >= bytes) return;
    if (b.buf) {
        vkDeviceWaitIdle(device_);
        destroy(b);
    }
    b = createBuffer(bytes);
}

void Context::transfer(Buffer& dev, bool toDevice, void* host, size_t bytes, size_t offset) {
    if (recording_) throw Error("upload/download while recording");
    if (bytes == 0) return;
    if (offset + bytes > dev.size) throw Error("transfer out of range");
    if (dev.mapped) {
        char* p = static_cast<char*>(dev.mapped) + offset;
        if (toDevice) std::memcpy(p, host, bytes); else std::memcpy(host, p, bytes);
        return;
    }
    if (!staging_.buf || staging_.size < bytes) {
        if (staging_.buf) { vkDeviceWaitIdle(device_); destroy(staging_); }
        staging_.size = (bytes + 15) & ~size_t(15);
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = staging_.size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(vkCreateBuffer(device_, &bci, nullptr, &staging_.buf), "vkCreateBuffer(staging)");
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, staging_.buf, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device_, &mai, nullptr, &staging_.mem), "vkAllocateMemory(staging)");
        check(vkBindBufferMemory(device_, staging_.buf, staging_.mem, 0), "vkBindBufferMemory(staging)");
        check(vkMapMemory(device_, staging_.mem, 0, VK_WHOLE_SIZE, 0, &staging_.mapped), "vkMapMemory(staging)");
    }
    if (toDevice) std::memcpy(staging_.mapped, host, bytes);
    begin();
    VkBufferCopy region{toDevice ? 0 : offset, toDevice ? offset : 0, bytes};
    if (toDevice) vkCmdCopyBuffer(cmd_, staging_.buf, dev.buf, 1, &region);
    else vkCmdCopyBuffer(cmd_, dev.buf, staging_.buf, 1, &region);
    submitWait();
    if (!toDevice) std::memcpy(host, staging_.mapped, bytes);
}

void Context::upload(Buffer& b, const void* data, size_t bytes, size_t offset) {
    transfer(b, true, const_cast<void*>(data), bytes, offset);
}

void Context::download(const Buffer& b, void* data, size_t bytes, size_t offset) {
    transfer(const_cast<Buffer&>(b), false, data, bytes, offset);
}

Context::Pipeline& Context::pipeline(const char* kernel, uint32_t nBindings) {
    std::string name = kernel;
    if (nativeAtomics && hasNativeVariant(kernel)) name += "_native";
    auto it = pipelines_.find(name);
    if (it != pipelines_.end()) {
        if (it->second.nBindings != nBindings) throw Error(name + ": binding count mismatch");
        return it->second;
    }
    const ShaderBlob* blob = findShader(name.c_str());
    if (!blob) throw Error("unknown kernel " + name);
    Pipeline p;
    p.nBindings = nBindings;
    std::vector<VkDescriptorSetLayoutBinding> binds(nBindings);
    for (uint32_t i = 0; i < nBindings; i++) {
        binds[i] = {};
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = nBindings;
    dli.pBindings = binds.data();
    check(vkCreateDescriptorSetLayout(device_, &dli, nullptr, &p.dsl), "vkCreateDescriptorSetLayout");
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &p.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    check(vkCreatePipelineLayout(device_, &pli, nullptr, &p.layout), "vkCreatePipelineLayout");
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = blob->bytes;
    smi.pCode = blob->code;
    VkShaderModule mod;
    check(vkCreateShaderModule(device_, &smi, nullptr, &mod), "vkCreateShaderModule");
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = p.layout;
    VkResult r = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpi, nullptr, &p.pipe);
    vkDestroyShaderModule(device_, mod, nullptr);
    check(r, ("vkCreateComputePipelines(" + name + ")").c_str());
    return pipelines_[name] = p;
}

void Context::begin() {
    if (recording_) throw Error("begin() while recording");
    check(vkResetDescriptorPool(device_, descPool_, 0), "vkResetDescriptorPool");
    check(vkResetCommandBuffer(cmd_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd_, &bi), "vkBeginCommandBuffer");
    recording_ = true;
}

void Context::barrier() {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    VkPipelineStageFlags st = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(cmd_, st, st, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void Context::dispatch(const char* kernel, const std::vector<const Buffer*>& bufs, const void* pc, uint32_t pcSize,
                       uint32_t gx, uint32_t gy, uint32_t gz) {
    if (!recording_) throw Error("dispatch() outside recording");
    if (gx == 0 || gy == 0 || gz == 0) return;
    Pipeline& p = pipeline(kernel, uint32_t(bufs.size()));
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = descPool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &p.dsl;
    VkDescriptorSet set;
    check(vkAllocateDescriptorSets(device_, &ai, &set), "vkAllocateDescriptorSets (too many dispatches in one submit?)");
    std::vector<VkDescriptorBufferInfo> infos(bufs.size());
    std::vector<VkWriteDescriptorSet> writes(bufs.size());
    for (size_t i = 0; i < bufs.size(); i++) {
        if (!bufs[i] || !bufs[i]->buf) throw Error(std::string(kernel) + ": unbound buffer at binding " + std::to_string(i));
        infos[i] = {bufs[i]->buf, 0, VK_WHOLE_SIZE};
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set;
        writes[i].dstBinding = uint32_t(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device_, uint32_t(writes.size()), writes.data(), 0, nullptr);
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
    if (pcSize) vkCmdPushConstants(cmd_, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
    vkCmdDispatch(cmd_, gx, gy, gz);
    barrier();
}

void Context::fill(Buffer& b, uint32_t value, VkDeviceSize bytes) {
    if (!recording_) throw Error("fill() outside recording");
    vkCmdFillBuffer(cmd_, b.buf, 0, bytes == VK_WHOLE_SIZE ? VK_WHOLE_SIZE : ((bytes + 3) & ~VkDeviceSize(3)), value);
    barrier();
}

void Context::copy(const Buffer& src, Buffer& dst, VkDeviceSize bytes) {
    if (!recording_) throw Error("copy() outside recording");
    if (bytes == 0) return;
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cmd_, src.buf, dst.buf, 1, &region);
    barrier();
}

int Context::probeFloatAtomics(uint32_t count) {
    Buffer b = createBuffer(8 * 4);
    begin();
    fill(b, 0);
    uint32_t pc0[2] = {count, 0}, pc1[2] = {count, 1};
    dispatch("atomic_probe", {&b}, pc0, 8, (count + 255) / 256);
    dispatch("atomic_probe", {&b}, pc1, 8, (count + 255) / 256);
    submitWait();
    float v[8];
    download(b, v, sizeof(v));
    destroy(b);
    int wrong = v[0] == float(count) ? 0 : 1;
    for (uint32_t k = 0; k < 7; k++) {
        uint32_t expect = count / 7 + (k < count % 7 ? 1 : 0);
        if (v[1 + k] != float(expect)) wrong++;
    }
    if (wrong && std::getenv("VKTRAIN_DEBUG"))
        std::fprintf(stderr, "atomic probe n=%u: subgroup sum %g, per-lane sums %g %g %g %g %g %g %g\n", count, v[0], v[1], v[2],
                     v[3], v[4], v[5], v[6], v[7]);
    return wrong;
}

void Context::submitWait() {
    if (!recording_) throw Error("submitWait() without begin()");
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};   // results visible to mapped host reads
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    check(vkEndCommandBuffer(cmd_), "vkEndCommandBuffer");
    recording_ = false;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    check(vkResetFences(device_, 1, &fence_), "vkResetFences");
    check(vkQueueSubmit(queue_, 1, &si, fence_), "vkQueueSubmit");
    check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
}

}  // namespace vkt
