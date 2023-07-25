use std::ptr;
use std::borrow::Cow;
use std::ffi::CStr;
use std::result::Result;
use std::mem::{self, MaybeUninit};

use ash::{vk, Entry, Instance, Device};
use bitflags::bitflags;

use crate::{
    ffi,
    error::VmaError,
    alloc_info::AllocationCreateInfo
};

bitflags! {
    pub struct AllocatorCreateFlags: u32 {
        const EXTERNALLY_SYNCHRONIZED = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT as u32;
        const KHR_DEDICATED_ALLOCATION = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT as u32;
        const KHR_BIND_MEMORY2 = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT as u32;
        const EXT_MEMORY_BUDGET = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT as u32;
        const AMD_DEVICE_COHERENT_MEMORY = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT as u32;
        const BUFFER_DEVICE_ADDRESS = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT as u32;
        const EXT_MEMORY_PRIORITY = ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT as u32;
    }
}

pub struct Allocation(ffi::VmaAllocation);

pub struct AllocationInfo(ffi::VmaAllocationInfo);

impl AllocationInfo {
    pub fn memory_type(&self) -> u32 {
        self.0.memoryType
    }

    pub fn device_memory(&self) -> vk::DeviceMemory {
        self.0.deviceMemory
    }

    pub fn offset(&self) -> vk::DeviceSize {
        self.0.offset
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.0.size
    }

    pub fn mapped_data(&self) -> *mut std::ffi::c_void {
        self.0.pMappedData
    }

    pub fn user_data(&self) -> *mut std::ffi::c_void {
        self.0.pUserData
    }

    pub fn name(&self) -> Cow<str> {
        unsafe { CStr::from_ptr(self.0.pName).to_string_lossy() }
    }
}

#[allow(dead_code)]
fn null_vk_fn() {
    panic!("VMA library called null function pointer")
}

macro_rules! null_vk_fn {
    () => {
        unsafe { mem::transmute(null_vk_fn as usize) }
    };
}

/// Vulkan Memory Allocator object
pub struct VmaAllocator(ffi::VmaAllocator);

unsafe impl Send for VmaAllocator {}

impl VmaAllocator {
    /// Create a new allocator object
    /// 
    /// # Safety
    /// 
    /// - Function pointers are given to the VMA library from the given [`Instance`] and [`Device`], so
    /// ensure that the instance and device outlive the allocator
    pub fn new(
        entry: &Entry,
        instance: &Instance,
        device: &Device,
        phys_dev: vk::PhysicalDevice,
        flags: AllocatorCreateFlags,
        vk_api_version: u32
    ) -> Result<Self, VmaError> {
        let vk_fns = ffi::VmaVulkanFunctions {
            vkGetInstanceProcAddr: entry.static_fn().get_instance_proc_addr,
            vkGetDeviceProcAddr: instance.fp_v1_0().get_device_proc_addr,
            vkGetPhysicalDeviceProperties: null_vk_fn!(),
            vkGetPhysicalDeviceMemoryProperties: null_vk_fn!(),
            vkAllocateMemory: null_vk_fn!(),
            vkFreeMemory: null_vk_fn!(),
            vkMapMemory: null_vk_fn!(),
            vkUnmapMemory: null_vk_fn!(),
            vkFlushMappedMemoryRanges: null_vk_fn!(),
            vkInvalidateMappedMemoryRanges: null_vk_fn!(),
            vkBindBufferMemory: null_vk_fn!(),
            vkBindImageMemory: null_vk_fn!(),
            vkGetBufferMemoryRequirements: null_vk_fn!(),
            vkGetImageMemoryRequirements: null_vk_fn!(),
            vkCreateBuffer: null_vk_fn!(),
            vkDestroyBuffer: null_vk_fn!(),
            vkCreateImage: null_vk_fn!(),
            vkDestroyImage: null_vk_fn!(),
            vkCmdCopyBuffer: null_vk_fn!(),
            vkGetBufferMemoryRequirements2KHR: null_vk_fn!(),
            vkGetImageMemoryRequirements2KHR: null_vk_fn!(),
            vkBindBufferMemory2KHR: null_vk_fn!(),
            vkBindImageMemory2KHR: null_vk_fn!(),
            vkGetPhysicalDeviceMemoryProperties2KHR: null_vk_fn!(),
            vkGetDeviceBufferMemoryRequirements: null_vk_fn!(),
            vkGetDeviceImageMemoryRequirements: null_vk_fn!()
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

    pub fn create_buffer(
        &self,
        create_info: &vk::BufferCreateInfo,
        alloc_info: &AllocationCreateInfo
    ) -> Result<(vk::Buffer, Allocation, AllocationInfo), VmaError> {
        unsafe {
            let mut buf = MaybeUninit::uninit();
            let mut allocation = MaybeUninit::uninit();
            let mut allocation_info = MaybeUninit::uninit();

            let res = ffi::vmaCreateBuffer(
                self.0,
                create_info,
                &alloc_info.0,
                buf.as_mut_ptr(),
                allocation.as_mut_ptr(),
                allocation_info.as_mut_ptr()
            );

            match res {
                vk::Result::SUCCESS => {
                    let buf = buf.assume_init();
                    let allocation = Allocation(allocation.assume_init());
                    let allocation_info = AllocationInfo(allocation_info.assume_init());

                    Ok((buf, allocation, allocation_info))
                },

                res => Err(VmaError::BufferCreationFailed(res))
            }
        }
    }

    pub fn create_image(
        &self,
        create_info: &vk::ImageCreateInfo,
        alloc_info: &AllocationCreateInfo
    ) -> Result<(vk::Image, Allocation, AllocationInfo), VmaError> {
        unsafe {
            let mut image = MaybeUninit::uninit();
            let mut allocation = MaybeUninit::uninit();
            let mut allocation_info = MaybeUninit::uninit();

            let res = ffi::vmaCreateImage(
                self.0,
                create_info,
                &alloc_info.0,
                image.as_mut_ptr(),
                allocation.as_mut_ptr(),
                allocation_info.as_mut_ptr()
            );

            match res {
                vk::Result::SUCCESS => {
                    let image = image.assume_init();
                    let allocation = Allocation(allocation.assume_init());
                    let allocation_info = AllocationInfo(allocation_info.assume_init());

                    Ok((image, allocation, allocation_info))
                },

                res => Err(VmaError::ImageCreationFailed(res))
            }
        }
    }
}