//! Library initialization

use std::ptr;
use std::mem::MaybeUninit;

use ash::{vk, Entry, Instance, Device};
use bitflags::bitflags;

use crate::{
    ffi,
    VmaAllocator,
    error::VmaError
};

bitflags! {
    /// Flags for created [`VmaAllocator`]
    pub struct AllocatorCreateFlags: u32 {
        /// Disables internal synchronization of the allocator
        /// 
        /// VMA by default includes internal mutexes for multithreaded access. Using this flag disables this internal
        /// synchronization. You should almost always use this flag, and wrap the allocator in a [`Mutex`](std::sync::Mutex)
        /// if multithreaded access is needed.
        /// 
        /// Note that [`VmaAllocator`](crate::VmaAllocator) is not `Sync`. So if for any reason you need to use internal
        /// synchronization, you will have to wrap the allocator in a newtype and `unsafe impl Sync` for it
        const EXTERNALLY_SYNCHRONIZED = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT as u32;
        
        /// Enables usage of the `VK_KHR_dedicated_allocation` extension. Only applicable with Vulkan 1.0 since the extension has
        /// been promoted to core in 1.1
        /// 
        /// With this flag, the allocator will automatically allocate certain buffers and images in dedicated [`vk::DeviceMemory`]
        /// blocks, if the driver suggests it may improve performance.
        /// 
        /// With Vulkan 1.0, this flag can only be used if the following device extensions were enabled:
        /// - `VK_KHR_get_memory_requirements2`
        /// - `VK_KHR_dedicated_allocation`
        /// 
        /// This flag is not required for versions > 1.0
        /// 
        /// <details>
        /// <summary><b>Original VMA Docs</b></summary>
        /// Enables usage of `VK_KHR_dedicated_allocation` extension.
        /// 
        /// The flag works only if `VmaAllocatorCreateInfo::vulkanApiVersion
        /// == VK_API_VERSION_1_0`. When it is `VK_API_VERSION_1_1`, the flag is ignored because the extension has been promoted to
        /// Vulkan 1.1.
        /// 
        /// Using this extension will automatically allocate dedicated blocks of memory for some buffers and images instead
        /// of suballocating place for them out of bigger memory blocks (as if you explicitly used `VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT` flag)
        /// when it is recommended by the driver. It may improve performance on some GPUs.
        /// 
        /// You may set this flag only if you found out that following device extensions are supported, you enabled them while creating Vulkan
        /// device passed as `VmaAllocatorCreateInfo::device`, and you want them to be used internally by this library:
        /// - `VK_KHR_get_memory_requirements2` (device extension)
        /// - `VK_KHR_dedicated_allocation` (device extension)
        /// 
        /// When this flag is set, you can experience following warnings reported by Vulkan validation layer. You can ignore them.
        /// > vkBindBufferMemory(): Binding memory to buffer 0x2d but vkGetBufferMemoryRequirements() has not been called on that buffer.
        /// </details>
        const KHR_DEDICATED_ALLOCATION = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT as u32;

        /// Enables usage of the `VK_KHR_bind_memory2` extension. Only applicable with Vulkan 1.0 since the extension has been
        /// promoted to core in 1.1
        /// 
        /// With Vulkan 1.0, this flag can only be used if the `VK_KHR_bind_memory2` extension was enabled
        /// 
        /// This flag is not required for versions > 1.0
        /// 
        /// <details>
        /// <summary><b>Original VMA Docs</b></summary>
        /// Enables usage of `VK_KHR_bind_memory2` extension.
        /// 
        /// The flag works only if `VmaAllocatorCreateInfo::vulkanApiVersion == VK_API_VERSION_1_0`. When it is `VK_API_VERSION_1_1`, the
        /// flag is ignored because the extension has been promoted to Vulkan 1.1.
        ///  
        /// You may set this flag only if you found out that this device extension is supported, you enabled it while creating Vulkan device
        /// passed as `VmaAllocatorCreateInfo::device`, and you want it to be used internally by this library.
        ///  
        /// The extension provides functions `vkBindBufferMemory2KHR` and `vkBindImageMemory2KHR`,
        /// which allow to pass a chain of `pNext` structures while binding. This flag is required if you use `pNext` parameter in `vmaBindBufferMemory2()`
        /// or `vmaBindImageMemory2()`.
        /// </details>
        const KHR_BIND_MEMORY2 = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT as u32;

        /// Enables usage of the `VK_EXT_memory_budget` extension
        /// 
        /// Allows you to query current memory usage and budget, which will be more accurate than the estimate that the
        /// library uses otherwise
        /// 
        /// This flag can only be used if the `VK_EXT_memory budget` extension, and it's dependency `VK_KHR_get_physical_device_properties2`,
        /// was enabled
        /// 
        /// `VK_KHR_get_physical_device_properties2` has been promoted to core in 1.1, and so does not need to be enabled for versions > 1.0
        /// 
        /// <details>
        /// <summary><b>Original VMA Docs</b></summary>
        /// Enables usage of `VK_EXT_memory_budget` extension.
        ///  
        /// You may set this flag only if you found out that this device extension is supported, you enabled it while creating Vulkan
        /// device passed as `VmaAllocatorCreateInfo::device`, and you want it to be used internally by this library, along with another
        /// instance extension `VK_KHR_get_physical_device_properties2`, which is required by it (or Vulkan 1.1, where this extension is promoted).
        ///  
        /// The extension provides query for current memory usage and budget, which will probably be more accurate than an estimation used
        /// by the library otherwise.
        /// </details>
        const EXT_MEMORY_BUDGET = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT as u32;

        /// Enables usage of the `VK_AMD_device_coherent_memory` extension
        /// 
        /// This flag can only be used if:
        /// - The `VK_AMD_device_coherent_memory` extension has been enabled
        /// - [`vk::PhysicalDeviceCoherentMemoryFeaturesAMD::device_coherent_memory`] is true
        /// 
        /// <details>
        /// <summary><b>Original VMA Docs</b></summary>
        /// Enables usage of `VK_AMD_device_coherent_memory` extension.
        /// 
        /// You may set this flag only if you:
        /// - found out that this device extension is supported and enabled it while creating Vulkan device passed as `VmaAllocatorCreateInfo::device`,
        /// - checked that `VkPhysicalDeviceCoherentMemoryFeaturesAMD::deviceCoherentMemory` is true and set it while creating the Vulkan device,
        /// - want it to be used internally by this library.
        /// 
        /// The extension and accompanying device feature provide access to memory types with `VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD` and
        /// `VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD` flags. They are useful mostly for writing breadcrumb markers - a common method for debugging
        /// GPU crash/hang/TDR.
        /// 
        /// When the extension is not enabled, such memory types are still enumerated, but their usage is illegal.\nTo protect from this error,
        /// if you don't create the allocator with this flag, it will refuse to allocate any memory or create a custom pool in such memory type,
        /// returning `VK_ERROR_FEATURE_NOT_PRESENT`.
        /// </details>
        const AMD_DEVICE_COHERENT_MEMORY = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT as u32;

        /// Enables usage of the buffer device address feature
        /// 
        /// When this flag has been set, you can create buffers with `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` with this library. The library adds the
        /// `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` during allocation wherever needed.
        /// 
        /// The buffer address can be retrieved using [`get_buffer_device_address()`](Device::get_buffer_device_address)
        /// 
        /// This flag can only be used if:
        /// - The `VK_KHR_buffer_device_address` extension has been enabled. This is not required for Vulkan >= 1.2, where it has been promoted to core
        /// - The [`vk::PhysicalDeviceBufferDeviceAddressFeatures::buffer_device_address`] device feature has been enabled
        /// 
        /// <details>
        /// <summary><b>Original VMA Docs</b></summary>
        /// Enables usage of "buffer device address" feature, which allows you to use function `vkGetBufferDeviceAddress*` to get raw GPU pointer
        /// to a buffer and pass it for usage inside a shader.
        /// 
        /// You may set this flag only if you:
        /// 1. (For Vulkan version < 1.2) Found as available and enabled device extension `VK_KHR_buffer_device_address`. This extension is promoted
        /// to core Vulkan 1.2.
        /// 2. Found as available and enabled device feature `VkPhysicalDeviceBufferDeviceAddressFeatures::bufferDeviceAddress`. When this flag is set,
        /// you can create buffers with `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` using VMA. The library automatically adds `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`
        /// to allocated memory blocks wherever it might be needed.
        /// 
        /// For more information, see documentation chapter \\ref enabling_buffer_device_address.
        /// </details>
        const BUFFER_DEVICE_ADDRESS = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT as u32;

        /// Enables usage of the `VK_EXT_memory_priority` extension
        /// 
        /// <details>
        /// <summary><b>Original VMA Docs</b></summary>
        /// Enables usage of `VK_EXT_memory_priority` extension in the library.
        /// 
        /// You may set this flag only if you found available and enabled this device extension, along with `VkPhysicalDeviceMemoryPriorityFeaturesEXT::memoryPriority
        /// == VK_TRUE`, while creating Vulkan device passed as `VmaAllocatorCreateInfo::device`.
        /// 
        /// When this flag is used, `VmaAllocationCreateInfo::priority` and `VmaPoolCreateInfo::priority` are used to set priorities of allocated Vulkan memory.
        /// Without it, these variables are ignored. A priority must be a floating-point value between 0 and 1, indicating the priority of the allocation relative
        /// to other memory allocations. Larger values are higher priority. The granularity of the priorities is implementation-dependent. It is automatically passed
        /// to every call to `vkAllocateMemory` done by the library using structure `VkMemoryPriorityAllocateInfoEXT`. The value to be used for default priority is 0.5.
        /// 
        /// For more details, see the documentation of the `VK_EXT_memory_priority` extension.
        /// </details>
        const EXT_MEMORY_PRIORITY = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT as u32;
    }
}

/// Information about a [`VmaAllocator`]
#[repr(transparent)]
pub struct AllocatorInfo(ffi::VmaAllocatorInfo);

impl AllocatorInfo {
    pub fn instance(&self) -> vk::Instance {
        self.0.instance
    }

    pub fn phys_dev(&self) -> vk::PhysicalDevice {
        self.0.physicalDevice
    }

    pub fn device(&self) -> vk::Device {
        self.0.device
    }
}

/// # Initialization and other basic functions
impl VmaAllocator {
    /// Create a new allocator object
    /// 
    /// # Safety
    /// 
    /// Ensure that the given `entry`, `instance` and `device` outlive the allocator
    pub fn new(
        entry: &Entry,
        instance: &Instance,
        device: &Device,
        phys_dev: vk::PhysicalDevice,
        flags: AllocatorCreateFlags,
        vk_api_version: u32
    ) -> Result<Self, VmaError> {
        let vk_fns = ffi::VmaVulkanFunctions {
            vkGetInstanceProcAddr: Some(entry.static_fn().get_instance_proc_addr),
            vkGetDeviceProcAddr: Some(instance.fp_v1_0().get_device_proc_addr),
            vkGetPhysicalDeviceProperties: None,
            vkGetPhysicalDeviceMemoryProperties: None,
            vkAllocateMemory: None,
            vkFreeMemory: None,
            vkMapMemory: None,
            vkUnmapMemory: None,
            vkFlushMappedMemoryRanges: None,
            vkInvalidateMappedMemoryRanges: None,
            vkBindBufferMemory: None,
            vkBindImageMemory: None,
            vkGetBufferMemoryRequirements: None,
            vkGetImageMemoryRequirements: None,
            vkCreateBuffer: None,
            vkDestroyBuffer: None,
            vkCreateImage: None,
            vkDestroyImage: None,
            vkCmdCopyBuffer: None,
            vkGetBufferMemoryRequirements2KHR: None,
            vkGetImageMemoryRequirements2KHR: None,
            vkBindBufferMemory2KHR: None,
            vkBindImageMemory2KHR: None,
            vkGetPhysicalDeviceMemoryProperties2KHR: None,
            vkGetDeviceBufferMemoryRequirements: None,
            vkGetDeviceImageMemoryRequirements: None
        };

        let create_info = ffi::VmaAllocatorCreateInfo {
            flags: flags.bits(),
            physicalDevice: phys_dev,
            device: device.handle(),
            preferredLargeHeapBlockSize: 0,
            pAllocationCallbacks: ptr::null(),
            pDeviceMemoryCallbacks: ptr::null(),
            pHeapSizeLimit: ptr::null(),
            pVulkanFunctions: &vk_fns,
            instance: instance.handle(),
            vulkanApiVersion: vk_api_version,
            pTypeExternalMemoryHandleTypes: ptr::null()
        };

        unsafe {
            let mut vma_alloc = MaybeUninit::uninit();

            let res = ffi::vmaCreateAllocator(&create_info, vma_alloc.as_mut_ptr());

            match res {
                vk::Result::SUCCESS => Ok(Self(vma_alloc.assume_init())),
                res => Err(VmaError::CreationFailed(res))
            }
        }
    }

    /// Gets information about this allocator (associated device, physical device and instance)
    pub fn info(&self) -> AllocatorInfo {
        unsafe {
            let mut info = MaybeUninit::uninit();

            ffi::vmaGetAllocatorInfo(self.0, info.as_mut_ptr());

            AllocatorInfo(info.assume_init())
        }
    }

    /// Gets the [`vk::PhysicalDeviceProperties`] of the associated physical device
    pub fn phys_dev_props(&self) -> &vk::PhysicalDeviceProperties {
        unsafe {
            let mut p_props = MaybeUninit::uninit();

            ffi::vmaGetPhysicalDeviceProperties(self.0, p_props.as_mut_ptr());

            &*p_props.assume_init()
        }
    }

    /// Gets the [`vk::PhysicalDeviceMemoryProperties`] of the associated physical device
    /// 
    /// Can be convenient to use this function instead of fetching it otherwise
    pub fn mem_props(&self) -> &vk::PhysicalDeviceMemoryProperties {
        unsafe {
            let mut p_props = MaybeUninit::uninit();

            ffi::vmaGetMemoryProperties(self.0, p_props.as_mut_ptr());

            &*p_props.assume_init()
        }
    }

    /// Gets the [`vk::MemoryPropertyFlags`] of a memory type given it's index
    /// 
    /// Can be convenient to use this function instead of fetching it otherwise
    pub fn mem_type_props(&self, mem_type_idx: u32) -> vk::MemoryPropertyFlags {
        unsafe {
            let mut flags = MaybeUninit::uninit();

            ffi::vmaGetMemoryTypeProperties(self.0, mem_type_idx, flags.as_mut_ptr());

            flags.assume_init()
        }
    }

    /// Inform the library of the current frame index while rendering. Should be called every frame
    /// 
    /// The library updates budget info when this function is called, avoiding the overhead of
    /// recalculating on every allocation
    pub fn set_current_frame_idx(&self, idx: u32) {
        unsafe {
            ffi::vmaSetCurrentFrameIndex(self.0, idx);
        }
    }
}

impl Drop for VmaAllocator {
    fn drop(&mut self) {
        unsafe {
            ffi::vmaDestroyAllocator(self.0);
        }
    }
}