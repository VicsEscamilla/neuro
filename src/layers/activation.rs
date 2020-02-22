use std::f32::consts::E;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone)]
pub enum Activation {
    Sigmoid = 1,
    Tanh = 2,
    ReLU = 3
}

pub fn sigmoid(x: &f32) -> f32 {
    1. / (1. + E.powf(-x))
}


pub fn sigmoid_prime(x: &f32) -> f32 {
    sigmoid(x) * (1. - sigmoid(x))
}


pub fn tanh(x: &f32) -> f32 {
    x.tanh()
}


pub fn tanh_prime(x: &f32) -> f32 {
    1. - x*x
}


pub fn relu(x: &f32) -> f32 {
    if *x > 0. {
        *x
    } else {
        0.
    }
}


pub fn relu_prime(x: &f32) -> f32 {
    if *x > 0. {
        1.
    } else {
        0.
    }
}
