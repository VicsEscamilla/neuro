mod activation;
mod dense;

#[cfg(feature="opencl")]
pub mod gpu;

use super::{Mtx, mtx, Layer};

pub use activation::Activation;
pub use activation::prime;
pub use dense::Dense;
