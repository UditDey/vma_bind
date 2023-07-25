use std::ptr;

use ash::vk;
use bitflags::bitflags;

use crate::ffi;

bitflags! {
    pub struct AllocationCreateFlags: u32 {
        const DEDICATED_MEMORY = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT as u32;
        const NEVER_ALLOCATE = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT as u32;
        const MAPPED = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT as u32;
        const UPPER_ADDRESS = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT as u32;
        const DONT_BIND = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DONT_BIND_BIT as u32;
        const WITHIN_BUDGET = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT as u32;
        const CAN_ALIAS = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT as u32;
        const HOST_ACCESS_SEQUENTIAL_WRITE = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT as u32;
        const HOST_ACCESS_RANDOM = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT as u32;
        const HOST_ACCESS_ALLOW_TRANSFER_INSTEAD = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT as u32;
        const STRATEGY_MIN_MEMORY = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT as u32;
        const STRATEGY_MIN_TIME = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT as u32;
        const STRATEGY_MIN_OFFSET = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT as u32;
        const STRATEGY_BEST_FIT = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT as u32;
        const STRATEGY_FIRST_FIT = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT as u32;
    }
}

pub enum MemoryUsage {
    Unknown,
    GpuLazilyAllocated,
    Auto,
    AutoPreferDevice,
    AutoPreferHost
}

pub struct AllocationCreateInfo(pub(super) ffi::VmaAllocationCreateInfo);

impl AllocationCreateInfo {
    pub fn new() -> Self {
        Self(ffi::VmaAllocationCreateInfo {
            flags: AllocationCreateFlags::empty().bits(),
            usage: ffi::VmaMemoryUsage::VMA_MEMORY_USAGE_UNKNOWN,
            requiredFlags: vk::MemoryPropertyFlags::empty(),
            preferredFlags: vk::MemoryPropertyFlags::empty(),
            memoryTypeBits: 0,
            pool: ptr::null_mut(),
            pUserData: ptr::null_mut(),
            priority: 1.0,
        })
    }

    pub fn flags(mut self, flags: AllocationCreateFlags) -> Self {
        self.0.flags = flags.bits();
        self
    }

    pub fn usage(mut self, usage: MemoryUsage) -> Self {
        self.0.usage = match usage {
            MemoryUsage::Unknown => ffi::VmaMemoryUsage::VMA_MEMORY_USAGE_UNKNOWN,
            MemoryUsage::GpuLazilyAllocated => ffi::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED,
            MemoryUsage::Auto => ffi::VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
            MemoryUsage::AutoPreferDevice => ffi::VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            MemoryUsage::AutoPreferHost => ffi::VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_HOST
        };

        self
    }

    pub fn required_flags(mut self, flags: vk::MemoryPropertyFlags) -> Self {
        self.0.requiredFlags = flags;
        self
    }

    pub fn preferred_flags(mut self, flags: vk::MemoryPropertyFlags) -> Self {
        self.0.preferredFlags = flags;
        self
    }

    pub fn memory_type_bits(mut self, bits: u32) -> Self {
        self.0.memoryTypeBits = bits;
        self
    }

    pub fn pool(self) -> Self {
        todo!()
    }

    pub fn user_data(mut self, data: *mut std::ffi::c_void) -> Self {
        self.0.pUserData = data;
        self
    }

    pub fn priority(mut self, priority: f32) -> Self {
        self.0.priority = priority;
        self
    }
}