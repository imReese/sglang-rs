//! Cross-platform kernel boundary for the Rust SGLang runtime.
//!
//! This crate starts with CPU reference implementations so runtime code can
//! depend on stable kernel semantics before CUDA, Metal, ROCm, or other native
//! backends are wired in.

use std::fmt;

pub mod cpu;
pub mod cublas;
pub mod cuda;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Metal,
    Rocm,
    Musa,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TopK {
    Fixed(usize),
    PerRow(Vec<usize>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelError {
    Shape(String),
    InvalidArgument(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(message) => write!(formatter, "kernel shape error: {message}"),
            Self::InvalidArgument(message) => {
                write!(formatter, "kernel invalid argument: {message}")
            }
        }
    }
}

impl std::error::Error for KernelError {}

pub type KernelResult<T> = Result<T, KernelError>;
