# neuro
Ultra naive classic neural network implementation in Rust

## Build

### No GPU support
    cargo build

### With OpenCL support
    cargo build --features opencl

## Examples

### XOR
    cargo run --example xor

### MNIST
    cargo run --example mnist
