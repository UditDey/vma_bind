use std::slice;
use std::ffi::CStr;
use std::borrow::Cow;
use std::error::Error;
use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};

use ash::vk;
use bitflags::bitflags;

use crate::{
    ffi,
    error::{VmaError, DefragError},
    VmaAllocator
};

/// Opaque handle to a memory allocation made by the allocator
/// 
/// This object is returned by functions that allocate memory and needs to be kept
/// around to free the allocation
#[repr(transparent)]
pub struct Allocation(ffi::VmaAllocation);

/// Parameters of an [`Allocation`] object
#[repr(transparent)]
pub struct AllocationInfo(ffi::VmaAllocationInfo);

impl AllocationInfo {
    /// Memory type index that this allocation was allocated from
    pub fn memory_type(&self) -> u32 {
        self.0.memoryType
    }

    /// Handle to the [`vk::DeviceMemory`] that the allocation was created from
    /// 
    /// The memory object can be shared by multiple allocations
    pub fn device_memory(&self) -> vk::DeviceMemory {
        self.0.deviceMemory
    }

    /// Offset (in bytes) in the [`vk::DeviceMemory`] object where the allocation is placed
    /// 
    /// The `(device_memory, offset)` pair is unique to this allocation. This offset can
    /// change during defragmentation
    pub fn offset(&self) -> vk::DeviceSize {
        self.0.offset
    }

    /// Size (in bytes) of this allocation
    pub fn size(&self) -> vk::DeviceSize {
        self.0.size
    }

    /// Mapped pointer to this allocation
    /// 
    /// Its returned as an `Option<NonNull<c_void>>` since the pointer may be null.
    /// It is only valid when the allocation was created with the
    /// [`AllocationCreateFlags::MAPPED`](AllocationCreateFlags::MAPPED)
    /// flag set, and null otherwise
    pub fn mapped_data(&self) -> Option<NonNull<std::ffi::c_void>> {
        NonNull::new(self.0.pMappedData)
    }

    /// Custom general purpose pointer that was passed with [`AllocationCreateInfo::user_data()`]
    pub fn user_data(&self) -> *mut std::ffi::c_void {
        self.0.pUserData
    }

    pub fn name(&self) -> Cow<str> {
        unsafe { CStr::from_ptr(self.0.pName).to_string_lossy() }
    }
}

bitflags! {
    /// Flags to be passed while creating a new allocation
    pub struct AllocationCreateFlags: u32 {
        /// Set this flag if the allocation should have its own memory block
        ///
        /// Use it for special, big resources, like fullscreen images used as attachments
        const DEDICATED_MEMORY = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT as u32;

        /// Set this flag to only try to allocate from existing [`vk::DeviceMemory`] blocks and never create new blocks
        /// 
        /// If a new allocation cannot be placed in any of the existing blocks, it fails with a
        /// [`vk::Result::ERROR_OUT_OF_DEVICE_MEMORY`] error
        const NEVER_ALLOCATE = ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT as u32;

        /// Set this flag to use memory that will be persistently mapped
        /// 
        /// The pointer to the allocation will be available through
        /// [`AllocationInfo::mapped_data()`](crate::allocator::AllocationInfo::mapped_data)
        /// 
        /// It is valid to use this flag for memory types that are not [`HOST_VISIBLE`](vk::MemoryPropertyFlags::HOST_VISIBLE)
        /// (which can happen if [`MemoryUsage::AutoPreferDevice`] is used). In that case the flag is ignored and the memory
        /// is not mapped (the mapped pointer will be null)
        /// 
        /// This behaviour is useful if you want to use device memory that is also preferably host mapped on platforms that
        /// support it (like integrated GPUs)
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

/// Intended usage of allocated memory
pub enum MemoryUsage {
    Unknown,
    GpuLazilyAllocated,
    Auto,
    AutoPreferDevice,
    AutoPreferHost
}

/// Parameters of a new allocation
#[repr(transparent)]
pub struct AllocationCreateInfo(ffi::VmaAllocationCreateInfo);

impl AllocationCreateInfo {
    /// Creates a new [`AllocationCreateInfo`] with default values
    /// 
    /// Provides a builder API to override defaults. You typically want to atleast set
    /// [`usage()`](Self::usage)
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

    /// Set allocation flags
    /// 
    /// Default value is [`AllocationCreateFlags::empty()`]
    pub fn flags(mut self, flags: AllocationCreateFlags) -> Self {
        self.0.flags = flags.bits();
        self
    }

    /// Set intended memory usage
    /// 
    /// Default value is [`MemoryUsage::Unknown`]. You typically want to override this value with
    /// [`Auto`](MemoryUsage::Auto), [`AutoPreferHost`](MemoryUsage::AutoPreferHost) or
    /// [`AutoPreferDevice`](MemoryUsage::AutoPreferDevice)
    /// 
    /// If a [`pool()`](Self::pool) has been set then this field is ignored
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

    /// Set [`vk::MemoryPropertyFlags`] that must be set in the memory type chosen for allocation
    /// 
    /// Default value is [`vk::MemoryPropertyFlags::empty()`], which means any memory type is
    /// acceptable
    /// 
    /// This is useful for special cases. Typically you want to use the [`usage()`](Self::usage)
    /// field with [`Auto`](MemoryUsage::Auto), [`AutoPreferHost`](MemoryUsage::AutoPreferHost) or
    /// [`AutoPreferDevice`](MemoryUsage::AutoPreferDevice) to select memory types instead
    /// 
    /// If a [`pool()`](Self::pool) has been set then this field is ignored
    pub fn required_flags(mut self, flags: vk::MemoryPropertyFlags) -> Self {
        self.0.requiredFlags = flags;
        self
    }

    /// Set [`vk::MemoryPropertyFlags`] that must preferrably be set in the memory type chosen for allocation
    /// 
    /// Default value is [`vk::MemoryPropertyFlags::empty()`], which means no memory type
    /// is preferred
    /// 
    /// This is useful for special cases. Typically you want to use the [`usage()`](Self::usage)
    /// field with [`Auto`](MemoryUsage::Auto), [`AutoPreferHost`](MemoryUsage::AutoPreferHost) or
    /// [`AutoPreferDevice`](MemoryUsage::AutoPreferDevice) to select memory types instead
    ///
    /// If a [`pool()`](Self::pool) has been set then this field is ignored
    pub fn preferred_flags(mut self, flags: vk::MemoryPropertyFlags) -> Self {
        self.0.preferredFlags = flags;
        self
    }

    /// Set bitmask containing one bit set for every memory type acceptable for this allocation
    /// 
    /// Default value is `0`, which is equivalent to [`u32::MAX`], which means that every memory
    /// type will be considered for this allocation
    /// 
    /// This is useful for special cases. Typically you want to use the [`usage()`](Self::usage)
    /// field with [`Auto`](MemoryUsage::Auto), [`AutoPreferHost`](MemoryUsage::AutoPreferHost) or
    /// [`AutoPreferDevice`](MemoryUsage::AutoPreferDevice) to select memory types instead
    ///
    /// If a [`pool()`](Self::pool) has been set then this field is ignored
    pub fn memory_type_bits(mut self, bits: u32) -> Self {
        self.0.memoryTypeBits = bits;
        self
    }

    /// Set a [`Pool`] that this allocation should be created in
    /// 
    /// Default value is `null`, which means that the default pool will be used
    /// 
    /// Specifying a pool means that allocations will be created only from the memory in that
    /// pool. Hence, all fields of this struct related to selecting memory types will be ignored.
    /// 
    /// See the [`Pool`] docs for more info on using custom memory pools
    pub fn pool(mut self, pool: &Pool) -> Self {
        self.0.pool = pool.0;
        self
    }

    /// Set a custom general purpose pointer that will be associated with this allocation
    /// 
    /// Default value is `null`
    /// 
    /// It can be retrieved by [`AllocationInfo::user_data()`] and set by [`VmaAllocator::set_alloc_user_data()`]
    pub fn user_data(mut self, data: *mut std::ffi::c_void) -> Self {
        self.0.pUserData = data;
        self
    }

    /// Set a floating point value between `0.0` and `1.0` indicating the priority of this allocation
    /// relative to other allocations
    /// 
    /// Default value is `1.0`
    /// 
    /// This field is used only if:
    /// - The [`AllocatorCreateFlags::EXT_MEMORY_PRIORITY`](super::init::AllocatorCreateFlags) flag was
    /// used while creating the allocator
    /// - This allocation ends up in a dedicated memory block, or is forced into a dedicated memory block
    /// using the [`AllocationCreateFlags::DEDICATED_MEMORY`] flag
    /// 
    /// Otherwise the allocation has the same priority as whichever memory block it was allocated from and
    /// this field has no impact
    pub fn priority(mut self, priority: f32) -> Self {
        self.0.priority = priority;
        self
    }
}

/// A custom VMA memory pool
#[repr(transparent)]
pub struct Pool(ffi::VmaPool);

bitflags! {
    pub struct PoolCreateFlags: u32 {
        const IGNORE_BUFFER_IMAGE_GRANULARITY = ffi::VmaPoolCreateFlagBits::VMA_POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT as u32;
        const LINEAR_ALGORITHM = ffi::VmaPoolCreateFlagBits::VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT as u32;
    }
}

/// Parameters of a [`Pool`]
#[repr(transparent)]
pub struct PoolCreateInfo(ffi::VmaPoolCreateInfo);

impl PoolCreateInfo {
    pub fn new(mem_type_idx: u32) -> Self {
        Self(ffi::VmaPoolCreateInfo {
            memoryTypeIndex: mem_type_idx,
            flags: PoolCreateFlags::empty().bits(),
            blockSize: 0,
            minBlockCount: 0,
            maxBlockCount: 0,
            priority: 1.0,
            minAllocationAlignment: 0,
            pMemoryAllocateNext: ptr::null_mut()
        })
    }
}

/// Defragmentation algorithm modes
/// 
/// This affects the defragmentation speed and number of moves to be made
pub enum DefragAlgorithm {
    Fast,
    Balanced,
    Full,

    /// Only available when [`buffer_image_granularity`](vk::PhysicalDeviceLimits::buffer_image_granularity)
    /// is greater than 1, since it aims to reduce alignment issues between different resource types.
    /// Otherwise falls back to same behaviour as `Full`
    Extensive
}

/// Configuration for defragmentation
#[repr(transparent)]
pub struct DefragInfo(ffi::VmaDefragmentationInfo);

impl DefragInfo {
    /// Create a new [`DefragInfo`] with default values
    /// 
    /// Provides a builder API to override defaults
    pub fn new() -> Self {
        Self(ffi::VmaDefragmentationInfo {
            flags: ffi::VmaDefragmentationFlagBits::VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT as u32,
            pool: ptr::null_mut(),
            maxBytesPerPass: 0,
            maxAllocationsPerPass: 0
        })
    }

    /// Set the algorithm mode
    /// 
    /// Default value is [`Balanced`](DefragAlgorithm::Balanced)
    pub fn algo(mut self, algo: DefragAlgorithm) -> Self {
        self.0.flags = match algo {
            DefragAlgorithm::Fast => ffi::VmaDefragmentationFlagBits::VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT as u32,
            DefragAlgorithm::Balanced => ffi::VmaDefragmentationFlagBits::VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT as u32,
            DefragAlgorithm::Full => ffi::VmaDefragmentationFlagBits::VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT as u32,
            DefragAlgorithm::Extensive => ffi::VmaDefragmentationFlagBits::VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT as u32,
        };

        self
    }

    /// Set a custom memory [`Pool`] to be defragmented
    /// 
    /// Default value is `null`, which means only default pool will undergo defragmentation
    pub fn pool(mut self, pool: &Pool) -> Self {
        self.0.pool = pool.0;
        self
    }

    /// Set the maximum number of bytes that can be moved in a single defragmentation pass
    /// 
    /// Default value is `0`, which means no limit
    pub fn max_bytes_per_pass(mut self, max: u64) -> Self {
        self.0.maxBytesPerPass = max;
        self
    }

    /// Set the maximum number of allocation that can be moved in a single defragmentation pass
    /// 
    /// Default value is `0`, which means no limit
    pub fn max_allocs_per_pass(mut self, allocs: u32) -> Self {
        self.0.maxAllocationsPerPass = allocs;
        self
    }
}

/// Operation to be performed on an allocation at the end of a defragmentation pass
pub enum DefragMoveOp {
    /// This means that the buffer/image has been recreated at `dst_tmp_alloc`, data has been copied
    /// the old buffer/image has been destroyed and `src_alloc` should be changed to point to the
    /// new place
    /// 
    /// This is the default value placed in every [`DefragMove`] at the start of a pass
    Copy,
    
    /// Does nothing with the allocation. Use this value if you cannot move the allocation. The new
    /// `dst_tmp_alloc` will be freed and `src_alloc` remains unchanged
    Ignore,

    /// Use this value if you want to abandon the allocation. Ensure that the buffer/image has been
    /// destroyed. Both `src_alloc` and `dst_tmp_alloc` will be freed
    Destroy
}

/// Describes an allocation that needs to be moved during defragmentation
/// 
/// Every defragmentation pass, VMA generates a list of [`DefragMove`]s, which is passed to the
/// `pass_handler`. [`src_alloc()`](Self::src_alloc) represents the original allocation, and
/// [`dst_tmp_alloc()`](Self::dst_tmp_alloc) represents the new location that the resource (and
/// its data) must be moved and bound to
/// 
/// So, in the typical case, the pass handler must:
/// - Copy the data from `src_alloc` to `dst_tmp_alloc`
/// - Destroy the buffer/image bound to `src_alloc`
/// - Recreate the buffer/image and bind it to `dst_tmp_alloc`
/// 
/// Every [`DefragMove`] additionally stores a [`DefragMoveOp`] operation. At the end of the pass,
/// the list of [`DefragMove`]s is passed back to VMA, which performs the specified operation
/// on each allocation. The default operation stored  at the start of the pass is
/// [`Copy`](DefragMoveOp::Copy), which represents the typical usage case. Use [`set_op`](Self::set_op)
/// to change the operation
#[repr(transparent)]
pub struct DefragMove(ffi::VmaDefragmentationMove);

impl DefragMove {
    /// Set the operation to be performed on this allocation at the end of the defragmentation pass
    /// 
    /// Default value is always [`Copy`](DefragMoveOp::Copy)
    pub fn set_op(&mut self, op: DefragMoveOp) {
        self.0.operation = match op {
            DefragMoveOp::Copy => ffi::VmaDefragmentationMoveOperation::VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY,
            DefragMoveOp::Ignore => ffi::VmaDefragmentationMoveOperation::VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE,
            DefragMoveOp::Destroy => ffi::VmaDefragmentationMoveOperation::VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY
        };
    }

    /// The original allocation that needs to be moved
    pub fn src_alloc(&self) -> Allocation {
        Allocation(self.0.srcAllocation)
    }

    /// Temporary allocation pointing to the location that will replace `src_alloc`
    pub fn dst_tmp_alloc(&self) -> Allocation {
        Allocation(self.0.dstTmpAllocation)
    }
}

#[repr(transparent)]
pub struct DefragStats(ffi::VmaDefragmentationStats);

impl DefragStats {
    pub fn bytes_moved(&self) -> vk::DeviceSize {
        self.0.bytesMoved
    }

    pub fn bytes_freed(&self) -> vk::DeviceSize {
        self.0.bytesFreed
    }

    pub fn allocs_moved(&self) -> u32 {
        self.0.allocationsMoved
    }

    pub fn blocks_freed(&self) -> u32 {
        self.0.deviceMemoryBlocksFreed
    }
}

/// # Allocation and pool related functions 
impl VmaAllocator {
    /// Helps to find a memory type index, given memory type bits and an [`AllocationCreateInfo`]
    /// 
    /// The algorithm finds a memory type that is:
    /// - Allowed by the memory type bits
    /// - Contains all the flags set by [`AllocationCreateInfo::required_flags()`]
    /// - Matches the usage set by [`AllocationCreateInfo::usage()`]
    /// - Contains as many flags set in [`AllocationCreateInfo::preferred_flags()`] as possible
    /// 
    /// Returns `None` if such a memory type is not found. In that case check the parameters of
    /// the resource, such as tiling ([`OPTIMAL`](vk::ImageTiling::OPTIMAL) or [`LINEAR`](vk::ImageTiling::LINEAR)),
    /// or mip level count
    /// 
    /// It can be useful, for example, to determine memory index to be passed to [`PoolCreateInfo::new()`]
    pub fn find_mem_type_idx(&self, mem_type_bits: u32, alloc_info: &AllocationCreateInfo) -> Option<u32> {
        unsafe {
            let mut idx = MaybeUninit::uninit();

            let res = ffi::vmaFindMemoryTypeIndex(self.0, mem_type_bits, &alloc_info.0, idx.as_mut_ptr());

            match res {
                vk::Result::SUCCESS => Some(idx.assume_init()),
                _ => None
            }
        }
    }

    /// Helps to find a memory type index, given [`vk::BufferCreateInfo`] and an [`AllocationCreateInfo`]
    /// 
    /// Similar to [`find_mem_type_idx()`](Self::find_mem_type_idx). It can be useful, for example,
    /// to determine memory index to be passed to [`PoolCreateInfo::new()`]. It internally creates a
    /// temporary dummy buffer that never has memory bound to it
    pub fn find_mem_type_idx_for_buf_info(
        &self,
        create_info: &vk::BufferCreateInfo,
        alloc_info: &AllocationCreateInfo
    ) -> Option<u32> {
        unsafe {
            let mut idx = MaybeUninit::uninit();

            let res = ffi::vmaFindMemoryTypeIndexForBufferInfo(self.0, create_info, &alloc_info.0, idx.as_mut_ptr());

            match res {
                vk::Result::SUCCESS => Some(idx.assume_init()),
                _ => None
            }
        }
    }

    /// Helps to find a memory type index, given [`vk::ImageCreateInfo`] and an [`AllocationCreateInfo`]
    /// 
    /// Similar to [`find_mem_type_idx()`](Self::find_mem_type_idx). It can be useful, for example,
    /// to determine memory index to be passed to [`PoolCreateInfo::new()`]. It internally creates a
    /// temporary dummy image that never has memory bound to it
    pub fn find_mem_type_idx_for_image_info(
        &self,
        create_info: &vk::ImageCreateInfo,
        alloc_info: &AllocationCreateInfo
    ) -> Option<u32> {
        unsafe {
            let mut idx = MaybeUninit::uninit();

            let res = ffi::vmaFindMemoryTypeIndexForImageInfo(self.0, create_info, &alloc_info.0, idx.as_mut_ptr());

            match res {
                vk::Result::SUCCESS => Some(idx.assume_init()),
                _ => None
            }
        }
    }

    /// Creates a custom memory [`Pool`]
    /// 
    /// See the [`Pool`] docs for more info on using custom memory pools
    pub fn create_pool(&self, create_info: &PoolCreateInfo) -> Result<Pool, VmaError> {
        unsafe {
            let mut pool = MaybeUninit::uninit();

            let res = ffi::vmaCreatePool(self.0, &create_info.0, pool.as_mut_ptr());

            match res {
                vk::Result::SUCCESS => Ok(Pool(pool.assume_init())),
                res => Err(VmaError::PoolCreationFailed(res))
            }
        }
    }

    /// Destroys a [`Pool`] and frees its memory
    pub fn destroy_pool(&self, pool: Pool) {
        unsafe {
            ffi::vmaDestroyPool(self.0, pool.0);
        }
    }

    /// Checks magic number in margins around all allocations in given memory pool in search for corruptions
    /// 
    /// Returns `Ok(())` if no corruption detected
    pub fn check_pool_corruption(&self, pool: &Pool) -> Result<(), VmaError> {
        unsafe {
            let res = ffi::vmaCheckPoolCorruption(self.0, pool.0);

            match res {
                vk::Result::SUCCESS => Ok(()),
                vk::Result::ERROR_UNKNOWN => Err(VmaError::CorruptionDetected),
                vk::Result::ERROR_FEATURE_NOT_PRESENT => Err(VmaError::CorruptionCheckNotEnabled),
                res => Err(VmaError::CorruptionCheckFailed(res))
            }
        }
    }

    /// Retrieves name of a [`Pool`] that was previously set with [`set_pool_name()`](Self::set_pool_name)
    pub fn get_pool_name(&self, pool: &Pool) -> String {
        unsafe {
            let mut name = MaybeUninit::uninit();

            ffi::vmaGetPoolName(self.0, pool.0, name.as_mut_ptr());

            // Copy into a new String because the name pointer becomes invalid if the pool
            // is destroyed or the name is changed
            CStr::from_ptr(name.assume_init())
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Sets name of a [`Pool`]
    pub fn set_pool_name(&self, pool: &Pool, name: &CStr) {
        unsafe {
            ffi::vmaSetPoolName(self.0, pool.0, name.as_ptr());
        }
    }

    /// General purpose memory allocation
    /// 
    /// Remember to free the memory using [`free_memory()`](Self::free_memory) or
    /// [`free_memory_pages()`](Self::free_memory_pages)
    /// 
    /// Its recommended to use [`alloc_memory_for_buf()`](Self::alloc_memory_for_buf),
    /// [`alloc_memory_for_image()`](Self::alloc_memory_for_image),
    /// [`create_buf()`](Self::create_buf) or [`create_image()`](Self::create_image)
    /// instead of this
    pub fn alloc_memory(
        &self,
        create_info: &AllocationCreateInfo,
        req: &vk::MemoryRequirements
    ) -> Result<(Allocation, AllocationInfo), VmaError> {
        unsafe {
            let mut alloc = MaybeUninit::uninit();
            let mut info = MaybeUninit::uninit();

            let res = ffi::vmaAllocateMemory(self.0, req, &create_info.0, alloc.as_mut_ptr(), info.as_mut_ptr());

            match res {
                vk::Result::SUCCESS => Ok((Allocation(alloc.assume_init()), AllocationInfo(info.assume_init()))),
                res => Err(VmaError::AllocationFailed(res))
            }
        }
    }

    /// General purpose memory allocation for multiple allocations at once
    /// 
    /// Remember to free the memory using [`free_memory()`](Self::free_memory) or
    /// [`free_memory_pages()`](Self::free_memory_pages)
    /// 
    /// The word "pages" is just a suggestion to use this function to allocate pieces of memory needed for sparse
    /// binding. It is just a general purpose allocation function able to make multiple allocations at once.
    /// It may be internally optimized to be more efficient than calling [`alloc_memory()`](Self::alloc_memory)
    /// multiple times
    /// 
    /// All allocations are made using same parameters. All of them are created out of the same memory pool and type.
    /// If any allocation fails, all allocations already made within this function call are also freed
    /// 
    /// # Panics
    /// 
    /// Panics if `create_infos` and `reqs` don't have the same length
    pub fn alloc_memory_pages(
        &self,
        create_infos: &[AllocationCreateInfo],
        reqs: &[vk::MemoryRequirements]
    ) -> Result<(Vec<Allocation>, Vec<AllocationInfo>), VmaError> {
        unsafe {
            assert!(
                create_infos.len() == reqs.len(),
                "create_infos and reqs must have equal length (create_infos.len() == {}, reqs.len() == {})",
                create_infos.len(),
                reqs.len()
            );

            let count = create_infos.len();

            let mut allocs: Vec<Allocation> = Vec::with_capacity(count);
            let mut infos: Vec<AllocationInfo> = Vec::with_capacity(count);

            // These casts are valid because the newtypes are repr(transparent)
            let res = ffi::vmaAllocateMemoryPages(
                self.0,
                reqs.as_ptr(),
                create_infos.as_ptr() as *const ffi::VmaAllocationCreateInfo,
                count,
                allocs.as_mut_ptr() as *mut ffi::VmaAllocation,
                infos.as_mut_ptr() as *mut ffi::VmaAllocationInfo
            );

            allocs.set_len(count);
            infos.set_len(count);

            match res {
                vk::Result::SUCCESS => Ok((allocs, infos)),
                res => Err(VmaError::AllocationFailed(res))
            }
        }
    }

    /// Allocate memory suitable for a [`vk::Buffer`]
    /// 
    /// It only creates a [`Allocation`]. It needs to be bound to the buffer using
    /// [`bind_buf_memory()`](Self::bind_buf_memory)
    /// 
    /// This is for special cases. Use [`create_buf()`](Self::create_buf) instead
    /// 
    /// Remember to free the memory using [`free_memory()`](Self::free_memory)
    pub fn alloc_memory_for_buf(
        &self,
        buf: vk::Buffer,
        create_info: &AllocationCreateInfo
    ) -> Result<(Allocation, AllocationInfo), VmaError> {
        unsafe {
            let mut alloc = MaybeUninit::uninit();
            let mut info = MaybeUninit::uninit();

            let res = ffi::vmaAllocateMemoryForBuffer(self.0, buf, &create_info.0, alloc.as_mut_ptr(), info.as_mut_ptr());

            match res {
                vk::Result::SUCCESS => Ok((Allocation(alloc.assume_init()), AllocationInfo(info.assume_init()))),
                res => Err(VmaError::AllocationFailed(res))
            }
        }
    }

    /// Allocate memory suitable for a [`vk::Image`]
    ///
    /// It only creates a [`Allocation`]. It needs to be bound to the image using
    /// [`bind_image_memory()`](Self::bind_image_memory)
    /// 
    /// This is for special cases. Use [`create_image()`](Self::create_image) instead
    /// 
    /// Remember to free the memory using [`free_memory()`](Self::free_memory)
    pub fn alloc_memory_for_image(
        &self,
        image: vk::Image,
        create_info: &AllocationCreateInfo
    ) -> Result<(Allocation, AllocationInfo), VmaError> {
        unsafe {
            let mut alloc = MaybeUninit::uninit();
            let mut info = MaybeUninit::uninit();

            let res = ffi::vmaAllocateMemoryForImage(self.0, image, &create_info.0, alloc.as_mut_ptr(), info.as_mut_ptr());

            match res {
                vk::Result::SUCCESS => Ok((Allocation(alloc.assume_init()), AllocationInfo(info.assume_init()))),
                res => Err(VmaError::AllocationFailed(res))
            }
        }
    }

    /// Frees an [`Allocation`]
    pub fn free_memory(&self, alloc: Allocation) {
        unsafe {
            ffi::vmaFreeMemory(self.0, alloc.0);
        }
    }

    /// Frees multiple allocations
    /// 
    /// # Safety
    /// 
    /// Since this function does not consume the allocations, ensure that they are dropped and
    /// their bound resources are not used after calling this
    pub fn free_memory_pages(&self, allocs: &[Allocation]) {
        unsafe {
            ffi::vmaFreeMemoryPages(self.0, allocs.len(), allocs.as_ptr() as *const ffi::VmaAllocation);
        }
    }

    /// Gets the current info about an allocation
    /// 
    /// Avoid calling this too often. Note that this info is also returned by functions that
    /// create/allocate resources. This info can be reused as long as the parameters do not
    /// change (for example, after defragmentation)
    pub fn get_alloc_info(&self, alloc: &Allocation) -> AllocationInfo {
        unsafe {
            let mut info = MaybeUninit::uninit();

            ffi::vmaGetAllocationInfo(self.0, alloc.0, info.as_mut_ptr());

            AllocationInfo(info.assume_init())
        }
    }

    /// Sets the `user_data` associated with an allocation
    pub fn set_alloc_user_data(&self, alloc: &Allocation, data: *mut std::ffi::c_void) {
        unsafe {
            ffi::vmaSetAllocationUserData(self.0, alloc.0, data);
        }
    }

    /// Set the name associated with an allocation
    pub fn set_alloc_name(&self, alloc: &Allocation, name: &CStr) {
        unsafe {
            ffi::vmaSetAllocationName(self.0, alloc.0, name.as_ptr());
        }
    }

    /// Get the [`vk::MemoryPropertyFlags`] of an allocation
    pub fn get_alloc_mem_props(&self, alloc: &Allocation) -> vk::MemoryPropertyFlags {
        unsafe {
            let mut flags = MaybeUninit::uninit();

            ffi::vmaGetAllocationMemoryProperties(self.0, alloc.0, flags.as_mut_ptr());

            flags.assume_init()
        }
    }

    /// Maps an allocation's memory and returns a pointer to it
    /// 
    /// The mapping can fail if attempted on an allocation in memory that is not
    /// [`HOST_VISIBLE`](vk::MemoryPropertyFlags::HOST_VISIBLE), in which case `None` is returned.
    /// This is internally refcounted. So the same allocation can be mapped multiple times, but also
    /// has to be unmapped the same number of times
    /// 
    /// It can be preferable to instead persistently map the memory using
    /// [`AllocationCreateFlags::MAPPED`](AllocationCreateFlags::MAPPED) during allocation.
    /// In that case calling this function is not required and the pointer can be retrieved from the
    /// [`AllocationInfo`] struct returned during allocation
    /// 
    /// This function doesn't automatically flush or invalidate caches. If the allocation is made from
    /// a memory type that is not [`HOST_COHERENT`](vk::MemoryPropertyFlags::HOST_COHERENT), you also
    /// need to use [`invalidate_alloc()`](Self::invalidate_alloc) before reads and
    /// [`flush_alloc()`](Self::flush_alloc) after writes
    pub fn map_memory(&self, alloc: &Allocation) -> Option<NonNull<std::ffi::c_void>> {
        unsafe {
            let mut ptr = MaybeUninit::uninit();

            let res = ffi::vmaMapMemory(self.0, alloc.0, ptr.as_mut_ptr());

            match res {
                vk::Result::SUCCESS => NonNull::new(ptr.assume_init()),
                _ => None
            }
        }
    }

    /// Unmaps an allocation's memory, that was previously mapped by [`map_memory()`](Self::map_memory)
    /// 
    /// Since memory mapping is internally refcounted, this needs to be called as many times as
    /// [`map_memory()`](Self::map_memory) was called on a particular allocation
    pub fn unmap_memory(&self, alloc: &Allocation) {
        unsafe {
            ffi::vmaUnmapMemory(self.0, alloc.0);
        }
    }

    /// Flushes a region of memory of a given allocation
    /// 
    /// This needs to be called after writing to mapped allocations that were created in a memory type
    /// that is not [`HOST_COHERENT`](vk::MemoryPropertyFlags::HOST_COHERENT).
    /// [`unmap_memory()`](Self::unmap_memory) does not do this automatically
    /// 
    /// Note that on PC hardware, memory types that are `HOST_VISIBLE` are also `HOST_COHERENT`, so calling
    /// this function is not needed
    /// 
    /// `size` can be specified as [`vk::WHOLE_SIZE`] to flush the entire memory range
    pub fn flush_alloc(
        &self,
        alloc: &Allocation,
        offset: vk::DeviceSize,
        size: vk::DeviceSize
    ) -> Result<(), VmaError> {
        unsafe {
            let res = ffi::vmaFlushAllocation(self.0, alloc.0, offset, size);

            match res {
                vk::Result::SUCCESS => Ok(()),
                res => Err(VmaError::FlushFailed(res))
            }
        }
    }

    /// Invalidates a region of memory of a given allocation
    /// 
    /// This needs to be called before reading from mapped allocations that were created in a memory type
    /// that is not [`HOST_COHERENT`](vk::MemoryPropertyFlags::HOST_COHERENT).
    /// [`map_memory()`](Self::map_memory) does not do this automatically
    /// 
    /// Note that on PC hardware, memory types that are `HOST_VISIBLE` are also `HOST_COHERENT`, so calling this
    /// function is not needed
    /// 
    /// `size` can be specified as [`vk::WHOLE_SIZE`] to invalidate the entire memory range
    pub fn invalidate_alloc(
        &self,
        alloc: &Allocation,
        offset: vk::DeviceSize,
        size: vk::DeviceSize
    ) -> Result<(), VmaError> {
        unsafe {
            let res = ffi::vmaInvalidateAllocation(self.0, alloc.0, offset, size);

            match res {
                vk::Result::SUCCESS => Ok(()),
                res => Err(VmaError::InvalidateFailed(res))
            }
        }
    }

    /// Flushes a region of memory of the given allocations
    /// 
    /// Same as [`flush_alloc()`](Self::flush_alloc) but for multiple allocations at once
    /// 
    /// # Panics
    /// 
    /// Panics if `allocs`, `offsets` and `sizes` don't have the same length
    pub fn flush_allocs(
        &self,
        allocs: &[Allocation],
        offsets: &[vk::DeviceSize],
        sizes: &[vk::DeviceSize]
    ) -> Result<(), VmaError> {
        unsafe {
            assert!(
                allocs.len() == offsets.len() && allocs.len() == sizes.len(),
                "allocs, offsets and sizes must have equal length (allocs.len() == {}, offsets.len() == {}, sizes.len() == {})",
                allocs.len(),
                offsets.len(),
                sizes.len()
            );

            let count = allocs.len() as u32;

            let res = ffi::vmaFlushAllocations(
                self.0,
                count,
                allocs.as_ptr() as *const ffi::VmaAllocation,
                offsets.as_ptr(),
                sizes.as_ptr()
            );

            match res {
                vk::Result::SUCCESS => Ok(()),
                res => Err(VmaError::FlushFailed(res))
            }
        }
    }

    /// Invalidates a region of memory of the given allocations
    /// 
    /// Same as [`invaldate_alloc()`](Self::invalidate_alloc) but for multiple allocations at once
    /// 
    /// # Panics
    /// 
    /// Panics if `allocs`, `offsets` and `sizes` don't have the same length
    pub fn invalidate_allocs(
        &self,
        allocs: &[Allocation],
        offsets: &[vk::DeviceSize],
        sizes: &[vk::DeviceSize]
    ) -> Result<(), VmaError> {
        unsafe {
            assert!(
                allocs.len() == offsets.len() && allocs.len() == sizes.len(),
                "allocs, offsets and sizes must have equal length (allocs.len() == {}, offsets.len() == {}, sizes.len() == {})",
                allocs.len(),
                offsets.len(),
                sizes.len()
            );

            let count = allocs.len() as u32;

            let res = ffi::vmaInvalidateAllocations(
                self.0,
                count,
                allocs.as_ptr() as *const ffi::VmaAllocation,
                offsets.as_ptr(),
                sizes.as_ptr()
            );

            match res {
                vk::Result::SUCCESS => Ok(()),
                res => Err(VmaError::FlushFailed(res))
            }
        }
    }

    /// Checks magic number in margins around all allocations in given memory types (in both default
    /// and custom pools) in search for corruptions
    /// 
    /// Returns `Ok(())` if no corruption detected
    pub fn check_corruption(&self, memory_type_bits: u32) -> Result<(), VmaError> {
        unsafe {
            let res = ffi::vmaCheckCorruption(self.0, memory_type_bits);

            match res {
                vk::Result::SUCCESS => Ok(()),
                vk::Result::ERROR_UNKNOWN => Err(VmaError::CorruptionDetected),
                vk::Result::ERROR_FEATURE_NOT_PRESENT => Err(VmaError::CorruptionCheckNotEnabled),
                res => Err(VmaError::CorruptionCheckFailed(res))
            }
        }
    }

    /// Performs memory defragmentation
    /// 
    /// Continous allocations and deallocations will, over time, cause memory fragmentation,
    /// resulting in the library being unable to find sufficient contigious free memory, even though
    /// sufficient free memory exists (just scattered between allocations), or excessive memory usage.
    /// The memory must therefore be regularly defragmented to reclaim more contigious free space
    /// 
    /// Defragmentation requires some allocations to be moved to new locations, and corresponding
    /// buffers/images to be rebound to these new locations. VMA does not do all this automatically.
    /// Instead, it tells you which allocations need to be moved where and it is your responsibility
    /// to follow it
    /// 
    /// Defragmentation occurs in multiple passes. In each pass VMA generates a list of allocations
    /// that need to be moved to new locations. This function takes a `pass_handler` closure that is
    /// called for each pass. The `pass_handler` takes a list of [`DefragMove`]s, each describing an
    /// allocation that needs to be moved
    /// 
    /// To handle errors within the pass handler, it needs to returns an `Result<(), E>`. If the
    /// return value is `Err` then defragmentation is ended early and the function returns with
    /// `Err(DefragError::PassHandlerError(E))`
    ///
    /// The defragmentation process can be configured by the [`DefragInfo`] struct
    /// 
    /// The general usage of this function should look like:
    /// ```
    /// let defrag_info = DefragInfo::new();
    /// 
    /// vma_alloc.defrag(
    ///     &defrag_info,
    ///     |moves| {
    ///         for mov in moves {
    ///             // Find which buffer/image this allocation corresponds to
    ///             // You need to include some kind of tag in the allocation's `user_data` for this
    ///             // Lets assume this corresponds to a buffer
    ///             let alloc_info = vma_alloc.get_alloc_info(mov.src_alloc());
    ///             let buf = find_buf_from_user_data(alloc_info.user_data());
    /// 
    ///             // Recreate the buffer, binding it at the new `dst_tmp_alloc`
    ///             let new_buf = device.create_buffer(...);
    ///             vma_alloc.bind_buf_memory(mov.dst_tmp_alloc(), buf);
    /// 
    ///             // Issue a copy command to copy the buffer's data to its new location
    ///             device.cmd_copy_buffer(...);
    /// 
    ///             // Optionally change the move op
    ///             mov.set_op(...);
    ///         }
    /// 
    ///         // Submit the copy commands
    ///         device.queue_submit(...)?;
    ///         device.wait_for_fences(...)?;
    /// 
    ///         // Destroy the old buffers
    ///         device.destroy_buffer(...);
    /// 
    ///         // Update descriptors to point to the new buffers
    ///         update_descriptors(...);
    /// 
    ///         Ok(())
    ///     }
    /// );
    /// ```
    pub fn defrag<F, E>(&self, defrag_info: &DefragInfo, pass_handler: F) -> Result<DefragStats, DefragError<E>>
    where
        F: Fn(&mut [DefragMove]) -> Result<(), E>,
        E: Error
    {
        unsafe {
            // Begin defragmentation and create defragmentation context
            let mut ctx = MaybeUninit::uninit();
            let res = ffi::vmaBeginDefragmentation(self.0, &defrag_info.0, ctx.as_mut_ptr());

            if res == vk::Result::ERROR_FEATURE_NOT_PRESENT {
                return Err(DefragError::DefragNotEnabled);
            }

            let ctx = ctx.assume_init();

            // Go through each required defragmentation pass
            loop {
                // Begin defragmentation pass and get the pass move info
                let mut pass_info = MaybeUninit::uninit();
                let res = ffi::vmaBeginDefragmentationPass(self.0, ctx, pass_info.as_mut_ptr());

                // Defragmentation complete, break out
                if res == vk::Result::SUCCESS {
                    break;
                }

                let mut pass_info = pass_info.assume_init();

                // Run the pass handler
                let mut moves = slice::from_raw_parts_mut(pass_info.pMoves as *mut DefragMove, pass_info.moveCount as usize);
                let res = pass_handler(&mut moves);

                // Pass handler failed, end defrag and return
                if let Err(err) = res {
                    let _ = ffi::vmaEndDefragmentationPass(self.0, ctx, &mut pass_info);
                    ffi::vmaEndDefragmentation(self.0, ctx, ptr::null_mut());

                    return Err(DefragError::PassHandlerError(err));
                }

                // End defragmentation pass
                let res = ffi::vmaEndDefragmentationPass(self.0, ctx, &mut pass_info);

                // Defragmentation complete, break out
                if res == vk::Result::SUCCESS {
                    break;
                }
            }

            // End defragmentation and collect stats
            let mut stats = MaybeUninit::uninit();
            ffi::vmaEndDefragmentation(self.0, ctx, stats.as_mut_ptr());

            Ok(DefragStats(stats.assume_init()))
        }
    }

    /// Binds a buffer to an allocation
    /// 
    /// Use this if you want to allocate memory first and then bind seperately. Do not use
    /// [`Device::bind_buffer_memory()`](ash::Device::bind_buffer_memory). This function ensures
    /// that `bind_buffer_memory()` and `map_memory()` are not called simultaneously by different
    /// threads, which is illegal in Vulkan
    pub fn bind_buf_memory(&self, alloc: &Allocation, buf: vk::Buffer) -> Result<(), VmaError> {
        unsafe {
            let res = ffi::vmaBindBufferMemory(self.0, alloc.0, buf);

            match res {
                vk::Result::SUCCESS => Ok(()),
                res => Err(VmaError::MemoryBindFailed(res))
            }
        }
    }

    /// Binds a buffer to an allocation with additional parameters
    /// 
    /// `alloc_local_offset` is the offset within the allocation, which should typically be 0
    /// 
    /// If `next` is not `null`, then the `VK_KHR_bind_memory2` extension is used, and `next` points
    /// to a chain of structures extending [`vk::BindBufferMemoryInfo`](ash::vk::BindBufferMemoryInfo).
    /// This requires the allocator to have been created with
    /// [`AllocatorCreateFlags::KHR_BIND_MEMORY2`](crate::init::AllocatorCreateFlags::KHR_BIND_MEMORY2),
    /// or for the Vulkan version to be >= 1.1 (where the extension is promoted to core). If these
    /// conditions are not met and `next` is not `null` then the function fails
    pub fn bind_buf_memory2<T: vk::ExtendsBindBufferMemoryInfo>(
        &self,
        alloc: &Allocation,
        alloc_local_offset: vk::DeviceSize,
        buf: vk::Buffer,
        next: &T
    ) -> Result<(), VmaError> {
        unsafe {
            let res = ffi::vmaBindBufferMemory2(self.0, alloc.0, alloc_local_offset, buf, next as *const T as _);

            match res {
                vk::Result::SUCCESS => Ok(()),
                res => Err(VmaError::MemoryBindFailed(res))
            }
        }
    }

    /// Binds an image to an allocation
    /// 
    /// Use this if you want to allocate memory first and then bind seperately. Do not use
    /// [`Device::bind_image_memory()`](ash::Device::bind_image_memory). This function ensures
    /// that `bind_image_memory()` and `map_memory()` are not called simultaneously by different
    /// threads, which is illegal in Vulkan
    pub fn bind_image_memory(&self, alloc: &Allocation, image: vk::Image) -> Result<(), VmaError> {
        unsafe {
            let res = ffi::vmaBindImageMemory(self.0, alloc.0, image);

            match res {
                vk::Result::SUCCESS => Ok(()),
                res => Err(VmaError::MemoryBindFailed(res))
            }
        }
    }

    /// Binds an image to an allocation with additional parameters
    /// 
    /// `alloc_local_offset` is the offset within the allocation, which should typically be 0
    /// 
    /// If `next` is not `null`, then the `VK_KHR_bind_memory2` extension is used, and `next` points
    /// to a chain of structures extending [`vk::BindImageMemoryInfo`](ash::vk::BindImageMemoryInfo).
    /// This requires the allocator to have been created with
    /// [`AllocatorCreateFlags::KHR_BIND_MEMORY2`](crate::init::AllocatorCreateFlags::KHR_BIND_MEMORY2),
    /// or for the Vulkan version to be >= 1.1 (where the extension is promoted to core). If these
    /// conditions are not met and `next` is not `null` then the function fails
    pub fn bind_image_memory2<T: vk::ExtendsBindImageMemoryInfo>(
        &self,
        alloc: &Allocation,
        alloc_local_offset: vk::DeviceSize,
        image: vk::Image,
        next: &T
    ) -> Result<(), VmaError> {
        unsafe {
            let res = ffi::vmaBindImageMemory2(self.0, alloc.0, alloc_local_offset, image, next as *const T as _);

            match res {
                vk::Result::SUCCESS => Ok(()),
                res => Err(VmaError::MemoryBindFailed(res))
            }
        }
    }

    /// Creates a [`vk::Buffer`] with memory allocated and bound to it
    /// 
    /// Remember to destroy the buffer and allocation using [`destroy_buf()`](Self::destroy_buf)
    /// 
    /// If [`AllocatorCreateFlags::KHR_DEDICATED_ALLOCATION`](crate::init::AllocatorCreateFlags::KHR_DEDICATED_ALLOCATION)
    /// was used while creating the allocator, then the `VK_KHR_dedicated_allocation` extension is used to query
    /// if this allocation needs a dedicated allocation. If yes, and [`AllocationCreateFlags::NEVER_ALLOCATE`]
    /// is not set, then a dedicated allocation is created for this buffer, just like using
    /// [`AllocationCreateFlags::DEDICATED_MEMORY`]. This flag is not needed for Vulkan version >= 1.1 where
    /// `VK_KHR_dedicated_allocation` has been promoted to core
    pub fn create_buf(
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

    /// Creates a [`vk::Buffer`] with memory allocated and bound to it, with an additional
    /// alignment constraint
    /// 
    /// Similar to [`create_buf()`](Self::create_buf) but has a `min_align` parameter,
    /// specifying the minimum alignment to be used when placing the buffer in a memory
    /// block, which may be needed sometimes (OpenGL interop for example)
    pub fn create_buf_with_alignment(
        &self,
        create_info: &vk::BufferCreateInfo,
        alloc_info: &AllocationCreateInfo,
        min_align: vk::DeviceSize
    ) -> Result<(vk::Buffer, Allocation, AllocationInfo), VmaError> {
        unsafe {
            let mut buf = MaybeUninit::uninit();
            let mut allocation = MaybeUninit::uninit();
            let mut allocation_info = MaybeUninit::uninit();

            let res = ffi::vmaCreateBufferWithAlignment(
                self.0,
                create_info,
                &alloc_info.0,
                min_align,
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

    /// Creates a [`vk::Buffer`] and binds already allocated memory to it
    /// 
    /// This function makes it convenient to create aliasing buffers (ie: many buffers
    /// bound to one allocation). Use [`alloc_memory()`](Self::alloc_memory) to create the
    /// allocation
    pub fn create_aliasing_buf(
        &self,
        create_info: &vk::BufferCreateInfo,
        alloc: &Allocation
    ) -> Result<vk::Buffer, VmaError> {
        unsafe {
            let mut buf = MaybeUninit::uninit();

            let res = ffi::vmaCreateAliasingBuffer(self.0, alloc.0, create_info, buf.as_mut_ptr());

            match res {
                vk::Result::SUCCESS => Ok(buf.assume_init()),
                res => Err(VmaError::BufferCreationFailed(res))
            }
        }
    }

    /// Destroys a [`vk::Buffer`] and frees its allocated memory
    /// 
    /// # Safety
    /// 
    /// Since this function does not consume the buffer ([`vk::Buffer`] is `Copy`), ensure
    /// that it is dropped and not used after calling this
    pub fn destroy_buf(&self, buf: vk::Buffer, alloc: Allocation) {
        unsafe {
            ffi::vmaDestroyBuffer(self.0, buf, alloc.0);
        }
    }

    /// Creates a [`vk::Image`] with memory allocated and bound to it
    /// 
    /// Similar to [`create_buf()`](Self::create_buf) but for images
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

    /// Creates a [`vk::Image`] and binds already allocated memory to it
    /// 
    /// This function makes it convenient to create aliasing images (ie: many images
    /// bound to one allocation). Use [`alloc_memory()`](Self::alloc_memory) to create the
    /// allocation
    pub fn create_aliasing_image(
        &self,
        create_info: &vk::ImageCreateInfo,
        alloc: &Allocation
    ) -> Result<vk::Image, VmaError> {
        unsafe {
            let mut image = MaybeUninit::uninit();

            let res = ffi::vmaCreateAliasingImage(self.0, alloc.0, create_info, image.as_mut_ptr());

            match res {
                vk::Result::SUCCESS => Ok(image.assume_init()),
                res => Err(VmaError::ImageCreationFailed(res))
            }
        }
    }

    /// Destroys a [`vk::Image`] and frees its allocated memory
    /// 
    /// # Safety
    /// 
    /// Since this function does not consume the image ([`vk::Image`] is `Copy`), ensure
    /// that it is dropped and not used after calling this
    pub fn destroy_image(&self, image: vk::Image, alloc: Allocation) {
        unsafe {
            ffi::vmaDestroyImage(self.0, image, alloc.0);
        }
    }
}