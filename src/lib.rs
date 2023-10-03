//! Bindings to the Vulkan Memory Allocator (VMA) library

mod ffi;
mod error;
mod init;
mod alloc;

//mod allocator;
//mod alloc_info;
//mod budget;

//pub use allocator::{AllocatorCreateFlags, Allocation, AllocationInfo, VmaAllocator};
//pub use alloc_info::{AllocationCreateFlags, MemoryUsage, AllocationCreateInfo};
//pub use budget::{Budget, Statistics};

pub use error::{VmaError, DefragError};
pub use init::{AllocatorCreateFlags, AllocatorInfo};

pub use alloc::{
    AllocationCreateFlags,
    AllocationCreateInfo,
    Allocation,
    AllocationInfo,

    MemoryUsage,
    Pool,
    PoolCreateFlags,
    PoolCreateInfo,

    DefragInfo,
    DefragAlgorithm,
    DefragMove,
    DefragMoveOp,
    DefragStats
};

/// Vulkan Memory Allocator object
pub struct VmaAllocator(ffi::VmaAllocator);