use std::fmt;
use std::error::Error;

use ash::vk;

/// VMA allocator error type
#[derive(Debug)]
pub enum VmaError {
    /// Failed to create the Vulkan Memory Allocator object
    CreationFailed(vk::Result),

    /// Failed to create or allocate memory for a [`vk::Buffer`]
    BufferCreationFailed(vk::Result),

    /// Failed to create or allocate memory for a [`vk::Image`]
    ImageCreationFailed(vk::Result)
}

impl fmt::Display for VmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreationFailed(res) => {
                writeln!(f, "Failed to create VMA allocator instance ({res:?}: {res})")?;
            },

            Self::BufferCreationFailed(res) => {
                writeln!(f, "Failed to create or allocate memory for a `vk::Buffer` ({res:?}: {res})")?;
            },

            Self::ImageCreationFailed(res) => {
                writeln!(f, "Failed to create or allocate memory for a `vk::Image` ({res:?}: {res})")?;
            }
        }

        Ok(())
    }
}

impl Error for VmaError {}