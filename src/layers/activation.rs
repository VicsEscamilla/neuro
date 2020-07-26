#[derive(Clone)]
pub enum Activation {
    Sigmoid = 1,
    Tanh = 2,
    ReLU = 3,
}


pub fn function(a: &Activation) -> impl Fn(&f32)->f32 {
    match a {
        Activation::Sigmoid => sigmoid,
        Activation::Tanh => tanh,
        Activation::ReLU => relu
    }
}


pub fn prime(a: &Activation) -> impl Fn(&f32)->f32 {
    match a {
        Activation::Sigmoid => sigmoid_prime,
        Activation::Tanh => tanh_prime,
        Activation::ReLU => relu_prime
    }
}


fn sigmoid(x: &f32) -> f32 {
    1. / (1. + (-x).exp())
}


fn sigmoid_prime(x: &f32) -> f32 {
    let s = sigmoid(x);
    s * (1. - s)
}


fn tanh(x: &f32) -> f32 {
    x.tanh()
}


fn tanh_prime(x: &f32) -> f32 {
    let t = tanh(x);
    1. - t*t
}


fn relu(x: &f32) -> f32 {
    x.max(0.)
}


fn relu_prime(x: &f32) -> f32 {
    if *x < 0. {
        0.
    } else {
        1.
    }
}
