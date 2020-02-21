#[macro_use]
extern crate neuro;

extern crate npy;

use gnuplot::*;
use neuro::{Neuro, Activation, Mtx, Runtime};

use std::io::Read;

use npy::NpyData;

fn load_file(file: &str) -> Vec<f32> {
    let mut buf = vec![];
    std::fs::File::open(file).unwrap().read_to_end(&mut buf).unwrap();
    return NpyData::from_bytes(&buf).unwrap().to_vec();
}

fn main() {
    let train_x_original: Vec<f32> = load_file("data/mnist_train_X.npy");
    let train_y_original: Vec<f32> = load_file("data/mnist_train_y.npy");
    let test_x_original: Vec<f32> = load_file("data/mnist_test_X.npy");
    let test_y_original: Vec<f32> = load_file("data/mnist_test_y.npy");

    println!("train_X -> {}", train_x_original.len());
    println!("train_y -> {}", train_y_original.len());
    println!("test_X -> {}", test_x_original.len());
    println!("test_X -> {}", test_y_original.len());

    let train_x = mtx![(5000, 784); &train_x_original[0..5000*784]];
    let train_y = mtx![(5000, 10); &train_y_original[0..5000*10]];
    let test_x = mtx![(1000, 784); &test_x_original[0..1000*784]];
    let test_y = mtx![(1000, 10); &test_y_original[0..1000*10]];

    let mut mnist_epochs: Vec<u64> = vec![];
    let mut mnist_train_mse: Vec<f32> = vec![];
    let mut mnist_test_mse: Vec<f32> = vec![];
    let mut fg = Figure::new();

    let mut digit = Neuro::new(Runtime::GPU)
        .add_layer(30, Activation::Sigmoid)
        .add_layer(10, Activation::Sigmoid)
        .on_epoch(move |epoch, total_epochs, train_mse, test_mse| {
            println!("epoch {} of {} -> train_mse: {}, test_mse: {}",
                epoch, total_epochs, train_mse, test_mse);
            mnist_epochs.push(epoch);
            mnist_train_mse.push(train_mse);
            mnist_test_mse.push(test_mse);
            fg.clear_axes();
            fg.axes2d()
                .set_title("MNIST - loss", &[])
                .set_legend(Graph(0.5), Graph(0.9), &[], &[])
                .set_x_label("epochs", &[])
                .set_y_label("loss", &[])
                .lines(mnist_epochs.iter(), mnist_train_mse.iter(), &[Caption("Train MSE")])
                .lines(mnist_epochs.iter(), mnist_test_mse.iter(), &[Caption("Test MSE")]);
            fg.show().unwrap();
        })
        .train(&train_x, &train_y, &test_x, &test_y, 3.0, 300000, 100);

    let mut successes = 0.;
    let total_tests = test_x.shape().0;
    println!("Evaluating...");
    for i in 0..total_tests {
        let expected = &test_y.get_row(i).get_raw();
        let got = digit.predict(&test_x.get_row(i)).unwrap().get_raw();

        let mut bigger_index = 0;
        for i in 0..got.len() {
            if &got[i] > &got[bigger_index] {
                bigger_index = i;
            }
        }

        successes += expected[bigger_index];
    }
    println!("Accuracy: {}", successes / total_tests as f32);
}
