/// @file vulkan.cpp
/// @brief Optional native Vulkan compute, explicit device buffers and SPIR-V pipelines.
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "gpu.hpp"
#ifdef DATAMUNGE_ENABLE_VULKAN
#include <vulkan/vulkan.h>
namespace datamunge::compute::detail {
/// @brief Private native driver resource and execution helpers.
namespace {
/// @brief Translate a failed Vulkan result, preserving the originating operation.
void check(VkResult result, const char* operation) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(
        std::string(operation) + " failed (VkResult " + std::to_string(int(result)) + ")");
}
/// @brief Pad a logical buffer/transfer size to Vulkan's four-byte copy granularity.
std::size_t padded(std::size_t n) {
  if (n > std::numeric_limits<std::size_t>::max() - 3)
    throw std::overflow_error("Vulkan buffer size overflow");
  return std::max(std::size_t(4), (n + 3) & ~std::size_t(3));
}
/// @brief Native buffer/memory pair; outer public resource ownership retains the device.
struct VulkanMemory final : GpuAllocation {
  VkDevice              device;                 ///< Allocating logical device.
  VkBuffer              buffer{VK_NULL_HANDLE}; ///< Storage/transfer buffer.
  VkDeviceMemory        memory{VK_NULL_HANDLE}; ///< Bound memory allocation.
  VkDeviceSize          size{0};                ///< Padded buffer length.
  VkMemoryPropertyFlags properties{0};          ///< Selected memory-type properties.

  /// @brief Retain the logical device that owns buffer memory.
  explicit VulkanMemory(VkDevice owner) : device(owner) {}
  /// @brief Destroy the buffer before releasing bound memory.
  ~VulkanMemory() {
    if (buffer)
      vkDestroyBuffer(device, buffer, nullptr);
    if (memory)
      vkFreeMemory(device, memory, nullptr);
  }
};
/// @brief Compiled shader, pipeline and reusable descriptor set owned by one device.
struct VulkanProgram final : GpuProgram {
  VkDevice              device;                      ///< Retained by public program context.
  VkShaderModule        shader{VK_NULL_HANDLE};      ///< Compiled SPIR-V module.
  VkDescriptorSetLayout descriptors{VK_NULL_HANDLE}; ///< Sequential storage-buffer layout.
  VkPipelineLayout      layout{VK_NULL_HANDLE};      ///< Descriptor and push-constant contract.
  VkPipeline            pipeline{VK_NULL_HANDLE};    ///< Native compute pipeline.
  VkDescriptorPool      pool{VK_NULL_HANDLE};        ///< Single reusable descriptor-set allocation.
  VkDescriptorSet       set{
      VK_NULL_HANDLE}; ///< Bound program arguments, updated between synchronous dispatches.

  /// @brief Retain the logical device owning pipeline resources.
  explicit VulkanProgram(VkDevice owner) : device(owner) {}
  /// @brief Release the complete native pipeline and descriptor resources.
  ~VulkanProgram() {
    if (pipeline)
      vkDestroyPipeline(device, pipeline, nullptr);
    if (pool)
      vkDestroyDescriptorPool(device, pool, nullptr);
    if (layout)
      vkDestroyPipelineLayout(device, layout, nullptr);
    if (descriptors)
      vkDestroyDescriptorSetLayout(device, descriptors, nullptr);
    if (shader)
      vkDestroyShaderModule(device, shader, nullptr);
  }
};
/// @brief Native Vulkan compute owner with independent resources and no window/surface dependency.
class VulkanDevice final : public GpuDevice {
  VkInstance       instance_{VK_NULL_HANDLE}; ///< Owned Vulkan instance.
  VkPhysicalDevice physical_{VK_NULL_HANDLE}; ///< Selected physical compute device.
  VkDevice         device_{VK_NULL_HANDLE};   ///< Exclusive logical device.
  VkQueue          queue_{VK_NULL_HANDLE};    ///< Compute/transfer queue, serialized by Context.
  std::uint32_t    family_{0};                ///< Compute queue-family index.
  VkCommandPool    commands_{VK_NULL_HANDLE}; ///< Reusable command-buffer pool.
  VkCommandBuffer  command_{VK_NULL_HANDLE};  ///< Reused synchronous command buffer.
  VkFence          fence_{VK_NULL_HANDLE};    ///< Reused submission fence.
  VkPhysicalDeviceProperties       properties_{}; ///< Device limits/name.
  VkPhysicalDeviceMemoryProperties memory_{};     ///< Available memory heaps/types.
 public:
  /// @brief Create a Vulkan 1.3 compute context, enabling supported numeric shader features.
  explicit VulkanDevice(std::uint32_t index) {
    try {
      VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
      application.pApplicationName = "datamunge compute";
      application.apiVersion       = VK_API_VERSION_1_3;
      VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
      create.pApplicationInfo = &application;
      check(vkCreateInstance(&create, nullptr, &instance_), "vkCreateInstance");
      std::uint32_t count = 0;
      check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
      std::vector<VkPhysicalDevice> devices(count);
      check(
          vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
          "vkEnumeratePhysicalDevices");
      if (index >= devices.size())
        throw std::runtime_error("Vulkan device index unavailable");
      physical_ = devices[index];
      vkGetPhysicalDeviceProperties(physical_, &properties_);
      if (properties_.apiVersion < VK_API_VERSION_1_3)
        throw std::runtime_error("Vulkan compute requires device API 1.3");
      vkGetPhysicalDeviceMemoryProperties(physical_, &memory_);
      std::uint32_t families = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, nullptr);
      std::vector<VkQueueFamilyProperties> queues(families);
      vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, queues.data());
      bool found = false;
      for (std::uint32_t i = 0; i < families; ++i)
        if (queues[i].queueCount && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
          family_ = i;
          found   = true;
          break;
        }
      if (!found)
        throw std::runtime_error("no Vulkan compute queue");
      float                   priority = 1;
      VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
      queue.queueFamilyIndex = family_;
      queue.queueCount       = 1;
      queue.pQueuePriorities = &priority;
      VkPhysicalDeviceFeatures supported{};
      vkGetPhysicalDeviceFeatures(physical_, &supported);
      VkPhysicalDeviceFeatures enabled{};
      enabled.shaderFloat64 = supported.shaderFloat64;
      enabled.shaderInt64   = supported.shaderInt64;
      enabled.shaderInt16   = supported.shaderInt16;
      VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
      device.queueCreateInfoCount = 1;
      device.pQueueCreateInfos    = &queue;
      device.pEnabledFeatures     = &enabled;
      check(vkCreateDevice(physical_, &device, nullptr, &device_), "vkCreateDevice");
      vkGetDeviceQueue(device_, family_, 0, &queue_);
      VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      pool.queueFamilyIndex = family_;
      pool.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      check(vkCreateCommandPool(device_, &pool, nullptr, &commands_), "vkCreateCommandPool");
      VkCommandBufferAllocateInfo buffer{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      buffer.commandPool        = commands_;
      buffer.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      buffer.commandBufferCount = 1;
      check(vkAllocateCommandBuffers(device_, &buffer, &command_), "vkAllocateCommandBuffers");
      VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      check(vkCreateFence(device_, &fence, nullptr, &fence_), "vkCreateFence");
    } catch (...) {
      cleanup();
      throw;
    }
  }
  /// @brief Finish and release the logical device, command pool and instance.
  ~VulkanDevice() override { cleanup(); }
  /// @brief Release native objects in reverse dependency order after pending work completes.
  void cleanup() noexcept {
    if (device_) {
      (void)vkDeviceWaitIdle(device_);
      if (fence_)
        vkDestroyFence(device_, fence_, nullptr);
      if (commands_)
        vkDestroyCommandPool(device_, commands_, nullptr);
      vkDestroyDevice(device_, nullptr);
      device_ = VK_NULL_HANDLE;
    }
    if (instance_) {
      vkDestroyInstance(instance_, nullptr);
      instance_ = VK_NULL_HANDLE;
    }
  }
  std::string name() const override { return properties_.deviceName; }
  /// @brief Allocate a padded Vulkan buffer with device-local or host-visible memory requirements.
  std::shared_ptr<VulkanMemory> memory(std::size_t bytes, bool staging) {
    auto result  = std::make_shared<VulkanMemory>(device_);
    result->size = padded(bytes);
    VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer.size  = result->size;
    buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   (staging ? 0 : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device_, &buffer, nullptr, &result->buffer), "vkCreateBuffer");
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device_, result->buffer, &requirements);
    int                   selected = -1;
    VkMemoryPropertyFlags required =
        staging ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    for (std::uint32_t i = 0; i < memory_.memoryTypeCount; ++i)
      if ((requirements.memoryTypeBits & (1u << i)) &&
          ((memory_.memoryTypes[i].propertyFlags & required) == required)) {
        selected = int(i);
        if (!staging ||
            (memory_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
          break;
      }
    if (selected < 0)
      throw std::runtime_error("Vulkan device lacks required buffer memory type");
    result->properties = memory_.memoryTypes[selected].propertyFlags;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize  = requirements.size;
    allocation.memoryTypeIndex = selected;
    check(vkAllocateMemory(device_, &allocation, nullptr, &result->memory), "vkAllocateMemory");
    check(vkBindBufferMemory(device_, result->buffer, result->memory, 0), "vkBindBufferMemory");
    return result;
  }
  std::shared_ptr<GpuAllocation> allocate(std::size_t bytes) override {
    return memory(bytes, false);
  }
  /// @brief Resolve a buffer view belonging to this exact logical device.
  VulkanMemory& get(const GpuArgument& argument) {
    auto* memory = dynamic_cast<VulkanMemory*>(argument.allocation.get());
    if (!memory || memory->device != device_ || argument.offset > memory->size ||
        argument.size > memory->size - argument.offset)
      throw std::invalid_argument("foreign or invalid Vulkan buffer view");
    return *memory;
  }
  /// @brief Record, submit and wait for one native command sequence.
  void execute(const std::function<void(VkCommandBuffer)>& record) {
    check(vkResetCommandBuffer(command_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command_, &begin), "vkBeginCommandBuffer");
    record(command_);
    check(vkEndCommandBuffer(command_), "vkEndCommandBuffer");
    check(vkResetFences(device_, 1, &fence_), "vkResetFences");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &command_;
    check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");
    check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  }
  /// @brief Map a staging allocation, flush/invalidate noncoherent memory and copy host bytes.
  void mapped(VulkanMemory& staging, void* bytes, std::size_t count, bool write) {
    void* address = nullptr;
    check(vkMapMemory(device_, staging.memory, 0, VK_WHOLE_SIZE, 0, &address), "vkMapMemory");
    try {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = staging.memory;
      range.size   = VK_WHOLE_SIZE;
      if (!write && !(staging.properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        check(vkInvalidateMappedMemoryRanges(device_, 1, &range), "vkInvalidateMappedMemoryRanges");
      if (write)
        std::memcpy(address, bytes, count);
      else
        std::memcpy(bytes, address, count);
      if (write && !(staging.properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        check(vkFlushMappedMemoryRanges(device_, 1, &range), "vkFlushMappedMemoryRanges");
    } catch (...) {
      vkUnmapMemory(device_, staging.memory);
      throw;
    }
    vkUnmapMemory(device_, staging.memory);
  }
  void upload(const GpuArgument& dst, const void* src, std::size_t bytes) override {
    if (!bytes)
      return;
    auto& target = get(dst);
    if (dst.offset % 4 || bytes % 4) {
      std::size_t low = dst.offset & ~std::size_t(3), size = padded(dst.offset - low + bytes);
      std::vector<std::uint8_t> adjusted(size);
      GpuArgument               aligned{dst.allocation, low, size};
      download(adjusted.data(), aligned, size);
      std::memcpy(adjusted.data() + dst.offset - low, src, bytes);
      upload(aligned, adjusted.data(), size);
      return;
    }
    auto staging = memory(bytes, true);
    mapped(*staging, const_cast<void*>(src), bytes, true);
    execute([&](VkCommandBuffer command) {
      VkBufferCopy copy{0, dst.offset, bytes};
      vkCmdCopyBuffer(command, staging->buffer, target.buffer, 1, &copy);
    });
  }
  void download(void* dst, const GpuArgument& src, std::size_t bytes) override {
    if (!bytes)
      return;
    auto&       source = get(src);
    std::size_t low = src.offset & ~std::size_t(3), size = padded(src.offset - low + bytes);
    if (low + size > source.size)
      throw std::out_of_range("Vulkan aligned transfer bounds");
    auto staging = memory(size, true);
    execute([&](VkCommandBuffer command) {
      VkBufferCopy copy{low, 0, size};
      vkCmdCopyBuffer(command, source.buffer, staging->buffer, 1, &copy);
    });
    std::vector<std::uint8_t> result(size);
    mapped(*staging, result.data(), size, false);
    std::memcpy(dst, result.data() + src.offset - low, bytes);
  }
  void copy(const GpuArgument& dst, const GpuArgument& src, std::size_t bytes) override {
    if (!bytes)
      return;
    auto& d = get(dst);
    auto& s = get(src);
    if (dst.offset % 4 || src.offset % 4 || bytes % 4)
      throw std::invalid_argument(
          "Vulkan buffer copy offsets/length must be multiples of four bytes");
    if (dst.allocation == src.allocation) {
      if (dst.offset == src.offset)
        return;
      if ((dst.offset < src.offset && bytes > src.offset - dst.offset) ||
          (src.offset < dst.offset && bytes > dst.offset - src.offset))
        throw std::invalid_argument("overlapping GPU copies require a temporary buffer");
    }
    execute([&](VkCommandBuffer command) {
      VkBufferCopy copy{src.offset, dst.offset, bytes};
      vkCmdCopyBuffer(command, s.buffer, d.buffer, 1, &copy);
    });
  }
  void fill(const GpuArgument& dst, std::uint8_t byte) override {
    if (!dst.size)
      return;
    auto& d = get(dst);
    if (dst.offset % 4 || dst.size % 4) {
      std::vector<std::uint8_t> bytes(dst.size, byte);
      upload(dst, bytes.data(), bytes.size());
      return;
    }
    execute([&](VkCommandBuffer command) {
      vkCmdFillBuffer(command, d.buffer, dst.offset, dst.size, std::uint32_t(byte) * 0x01010101u);
    });
  }
  std::shared_ptr<GpuProgram> program(const KernelSource& source) override {
    if (source.spirv.size() < 5 || source.spirv[0] != 0x07230203u)
      throw std::invalid_argument("Vulkan program requires SPIR-V words");
    if (source.parameter_bytes % 4 ||
        source.parameter_bytes > properties_.limits.maxPushConstantsSize ||
        source.buffers > properties_.limits.maxPerStageDescriptorStorageBuffers ||
        source.buffers > properties_.limits.maxDescriptorSetStorageBuffers)
      throw std::invalid_argument("Vulkan program layout exceeds device limits");
    std::uint64_t lanes = 1;
    for (int i = 0; i < 3; ++i) {
      if (source.local_size[i] > properties_.limits.maxComputeWorkGroupSize[i])
        throw std::invalid_argument("Vulkan local size exceeds device limits");
      lanes *= source.local_size[i];
    }
    if (lanes > properties_.limits.maxComputeWorkGroupInvocations)
      throw std::invalid_argument("Vulkan local invocation limit");
    auto                     result = std::make_shared<VulkanProgram>(device_);
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader.codeSize = source.spirv.size() * 4;
    shader.pCode    = source.spirv.data();
    check(vkCreateShaderModule(device_, &shader, nullptr, &result->shader), "vkCreateShaderModule");
    std::vector<VkDescriptorSetLayoutBinding> bindings(source.buffers);
    for (std::uint32_t i = 0; i < source.buffers; ++i)
      bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo descriptors{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptors.bindingCount = source.buffers;
    descriptors.pBindings    = bindings.data();
    check(
        vkCreateDescriptorSetLayout(device_, &descriptors, nullptr, &result->descriptors),
        "vkCreateDescriptorSetLayout");
    VkPushConstantRange        range{VK_SHADER_STAGE_COMPUTE_BIT, 0, source.parameter_bytes};
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount = 1;
    layout.pSetLayouts    = &result->descriptors;
    if (source.parameter_bytes) {
      layout.pushConstantRangeCount = 1;
      layout.pPushConstantRanges    = &range;
    }
    check(
        vkCreatePipelineLayout(device_, &layout, nullptr, &result->layout),
        "vkCreatePipelineLayout");
    VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline.layout       = result->layout;
    pipeline.stage        = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline.stage.module = result->shader;
    pipeline.stage.pName  = source.entry.c_str();
    check(
        vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &result->pipeline),
        "vkCreateComputePipelines");
    if (source.buffers) {
      VkDescriptorPoolSize       size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, source.buffers};
      VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      pool.maxSets       = 1;
      pool.poolSizeCount = 1;
      pool.pPoolSizes    = &size;
      check(
          vkCreateDescriptorPool(device_, &pool, nullptr, &result->pool), "vkCreateDescriptorPool");
      VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      allocate.descriptorPool     = result->pool;
      allocate.descriptorSetCount = 1;
      allocate.pSetLayouts        = &result->descriptors;
      check(vkAllocateDescriptorSets(device_, &allocate, &result->set), "vkAllocateDescriptorSets");
    }
    return result;
  }
  void dispatch(
      GpuProgram&                     source,
      const std::vector<GpuArgument>& buffers,
      const void*                     parameters,
      std::size_t                     bytes,
      std::array<std::uint32_t, 3>    groups) override {
    auto* program = dynamic_cast<VulkanProgram*>(&source);
    if (!program || program->device != device_)
      throw std::invalid_argument("foreign Vulkan program");
    for (int i = 0; i < 3; ++i)
      if (groups[i] > properties_.limits.maxComputeWorkGroupCount[i])
        throw std::invalid_argument("Vulkan dispatch dimensions exceed device limits");
    std::vector<VkDescriptorBufferInfo> infos(buffers.size());
    std::vector<VkWriteDescriptorSet>   writes(buffers.size());
    for (std::size_t i = 0; i < buffers.size(); ++i) {
      auto& memory = get(buffers[i]);
      if (buffers[i].offset % properties_.limits.minStorageBufferOffsetAlignment ||
          buffers[i].size > properties_.limits.maxStorageBufferRange)
        throw std::invalid_argument("Vulkan storage view alignment/range exceeds device limits");
      infos[i]                  = {memory.buffer, buffers[i].offset, buffers[i].size};
      writes[i]                 = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[i].dstSet          = program->set;
      writes[i].dstBinding      = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo     = &infos[i];
    }
    if (!writes.empty())
      vkUpdateDescriptorSets(device_, writes.size(), writes.data(), 0, nullptr);
    execute([&](VkCommandBuffer command) {
      VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
      before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(
          command,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          0,
          1,
          &before,
          0,
          nullptr,
          0,
          nullptr);
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, program->pipeline);
      if (program->set)
        vkCmdBindDescriptorSets(
            command,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            program->layout,
            0,
            1,
            &program->set,
            0,
            nullptr);
      if (bytes)
        vkCmdPushConstants(
            command, program->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, parameters);
      vkCmdDispatch(command, groups[0], groups[1], groups[2]);
      VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_HOST_READ_BIT;
      vkCmdPipelineBarrier(
          command,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
          0,
          1,
          &after,
          0,
          nullptr,
          0,
          nullptr);
    });
  }
  void synchronize() override { check(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle"); }
};
} // namespace
std::shared_ptr<GpuDevice> make_vulkan(std::uint32_t index) {
  return std::make_shared<VulkanDevice>(index);
}
} // namespace datamunge::compute::detail
#else
namespace datamunge::compute::detail {
std::shared_ptr<GpuDevice> make_vulkan(std::uint32_t) {
  throw std::runtime_error("Vulkan backend was not enabled at build time");
}
} // namespace datamunge::compute::detail
#endif
