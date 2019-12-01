extern crate neuro;

extern crate npy;

use neuro::{Neuro, Activation, Mtx, Runtime};

use std::io::Read;
use npy::NpyData;

fn load_file(file: &str) -> Vec<f32> {
    let mut buf = vec![];
    std::fs::File::open(file).unwrap().read_to_end(&mut buf).unwrap();
    return NpyData::from_bytes(&buf).unwrap().to_vec();
}

fn main() {
    let train_x_original: Vec<f32> = load_file("data/mnist_train_X.npy")[0..5000*784].to_vec();
    let train_y_original: Vec<f32> = load_file("data/mnist_train_y.npy")[0..5000*10].to_vec();
    let test_x_original: Vec<f32> = load_file("data/mnist_test_X.npy")[0..1000*784].to_vec();
    let test_y_original: Vec<f32> = load_file("data/mnist_test_y.npy")[0..1000*10].to_vec();

    println!("train_X -> {}", train_x_original.len());
    println!("train_y -> {}", train_y_original.len());
    println!("test_X -> {}", test_x_original.len());
    println!("test_X -> {}", test_y_original.len());

    let train_x = Mtx::new((5000, 784), train_x_original);
    let train_y = Mtx::new((5000, 10), train_y_original);
    let test_x = Mtx::new((1000, 784), test_x_original);
    let test_y = Mtx::new((1000, 10), test_y_original);

    let mut digit = Neuro::new(Runtime::GPU)
        .add_layer(30, Activation::Sigmoid)
        .add_layer(10, Activation::Sigmoid)
        .on_epoch(|epoch, total_epochs, train_mse, test_mse| {
            println!("epoch {} of {} -> train_mse: {}, test_mse: {}",
                epoch, total_epochs, train_mse, test_mse);
        })
        .train(&train_x, &train_y, &test_x, &test_y, 3.0, 30, 100);

    let mut successes = 0;
    let total_tests = test_x.shape().0;
    println!("Evaluating...");
    for i in 0..total_tests {
        let expected = &test_y.get_row(i);
        let input = test_x.get_row(i);
        let got = digit.predict(&input).unwrap().func(|x| {
                if *x < 0.5 { 0. } else { 1. }
            });
        if got.add(&expected.func(|x| -x)).sum(0).get_raw()[0] == 0.0 {
            successes += 1;
        }
    }
    println!("Accuracy: {}", successes as f32 / total_tests as f32);
}
