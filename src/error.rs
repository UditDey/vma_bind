use std::fmt;
use std::error::Error;

use ash::vk;

/// VMA allocator error type
#[derive(Debug)]
pub enum VmaError {
    CreationFailed(vk::Result),
    PoolCreationFailed(vk::Result),

    CorruptionDetected,
    CorruptionCheckNotEnabled,
    CorruptionCheckFailed(vk::Result),

    AllocationFailed(vk::Result),
    FlushFailed(vk::Result),
    InvalidateFailed(vk::Result),

    MemoryBindFailed(vk::Result),

    BufferCreationFailed(vk::Result),
    ImageCreationFailed(vk::Result)
}

impl fmt::Display for VmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreationFailed(res) => {
                writeln!(f, "Failed to create VMA allocator instance ({res:?}: {res})")?;
            },

            Self::PoolCreationFailed(res) => {
                writeln!(f, "Failed to create a `Pool` ({res:?}: {res})")?;
            },

            Self::CorruptionDetected => {
                writeln!(f, "Memory corruption has been detected")?;
            },

            Self::CorruptionCheckNotEnabled => {
                writeln!(f, "Memory corruption feature has not enabled")?;
            },

            Self::CorruptionCheckFailed(res) => {
                writeln!(f, "Memory corruption check failed for unknown reason ({res:?}: {res})")?;
            }

            Self::AllocationFailed(res) => {
                writeln!(f, "Failed to allocate memory ({res:?}: {res})")?;
            },

            Self::FlushFailed(res) => {
                writeln!(f, "Failed to flush an allocation ({res:?}: {res})")?;
            },

            Self::InvalidateFailed(res) => {
                writeln!(f, "Failed to invalidate an allocation ({res:?}: {res})")?;
            },

            Self::MemoryBindFailed(res) => {
                writeln!(f, "Failed to bind memory ({res:?}: {res})")?;
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

/// Special error type for defragmentation
#[derive(Debug)]
pub enum DefragError<E: Error> {
    DefragNotEnabled,
    PassHandlerError(E)
}

impl<E: Error> fmt::Display for DefragError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DefragNotEnabled => {
                writeln!(f, "Defragmentation feature has not been enabled")?;
            },

            Self::PassHandlerError(err) => {
                writeln!(f, "Defragmentation pass handler failed with error:")?;
                writeln!(f, "{err}")?;
            },
        }

        Ok(())
    }
}

impl<E: Error> Error for DefragError<E> {}