mod activation;
mod dense;
mod softmax;

#[cfg(feature="opencl")]
pub mod gpu;

use super::{Mtx, mtx, Layer};

pub use activation::Activation;
pub use dense::Dense;
pub use softmax::Softmax;
