use std::f32::consts::E;

#[derive(Clone)]
pub enum Activation {
    Sigmoid = 1,
    SigmoidSimple = 2,
    Tanh = 3,
    ReLU = 4
}


pub fn function(a: &Activation) -> impl Fn(&f32)->f32 {
    match a {
        Activation::Sigmoid => sigmoid,
        Activation::SigmoidSimple => sigmoid,
        Activation::Tanh => tanh,
        Activation::ReLU => relu
    }
}


pub fn prime(a: &Activation) -> impl Fn(&f32)->f32 {
    match a {
        Activation::Sigmoid => sigmoid_prime,
        Activation::SigmoidSimple => sigmoid_prime_simple,
        Activation::Tanh => tanh_prime,
        Activation::ReLU => relu_prime
    }
}


fn sigmoid(x: &f32) -> f32 {
    1. / (1. + E.powf(-x))
}


fn sigmoid_prime_simple(x: &f32) -> f32 {
    x * (1. - x)
}


fn sigmoid_prime(x: &f32) -> f32 {
    sigmoid(x) * (1. - sigmoid(x))
}


fn tanh(x: &f32) -> f32 {
    x.tanh()
}


fn tanh_prime(x: &f32) -> f32 {
    1. - x*x
}


fn relu(x: &f32) -> f32 {
    x.max(0.)
}


fn relu_prime(x: &f32) -> f32 {
    if *x > 0. {
        1.
    } else {
        0.
    }
}
