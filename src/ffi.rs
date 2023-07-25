#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use ash::vk::*;

#[cfg(unix)]
include!(concat!(env!("OUT_DIR"), "/vma_ffi.rs"));

#[cfg(windows)]
include!(concat!(env!("OUT_DIR"), "\\vma_ffi.rs"));