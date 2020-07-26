#[macro_use]
extern crate neuro;

extern crate npy;

use gnuplot::*;
use neuro::{Neuro, Mtx};
use neuro::layers::{Activation, gpu::Dense};
use std::time::Instant;

use std::io::Read;

use npy::NpyData;

fn load_file(file: &str) -> Vec<f32> {
    let mut buf = vec![];
    std::fs::File::open(file).unwrap().read_to_end(&mut buf).unwrap();
    return NpyData::from_bytes(&buf).unwrap().to_vec();
}

fn main() {
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), "/examples/mnist/data");
    let train_x_original: Vec<f32> = load_file(&format!("{}/{}", path, "mnist_train_X.npy"));
    let train_y_original: Vec<f32> = load_file(&format!("{}/{}", path, "mnist_train_y.npy"));
    let test_x_original: Vec<f32> = load_file(&format!("{}/{}", path, "mnist_test_X.npy"));
    let test_y_original: Vec<f32> = load_file(&format!("{}/{}", path, "mnist_test_y.npy"));

    println!("train_X -> {}", train_x_original.len());
    println!("train_y -> {}", train_y_original.len());
    println!("test_X -> {}", test_x_original.len());
    println!("test_X -> {}", test_y_original.len());

    let train_x = mtx![(50000, 784); &train_x_original[0..50000*784]];
    let train_y = mtx![(50000, 10); &train_y_original[0..50000*10]];
    let test_x = mtx![(10000, 784); &test_x_original[0..10000*784]];
    let test_y = mtx![(10000, 10); &test_y_original[0..10000*10]];

    let mut mnist_epochs: Vec<u64> = vec![];
    let mut mnist_train_loss: Vec<f32> = vec![];
    let mut mnist_test_loss: Vec<f32> = vec![];
    let mut fg = Figure::new();

    let total = Instant::now();
    let mut now = Instant::now();
    let mut digit = Neuro::new()
        .add_layer(Dense::new(30, Activation::Sigmoid))
        .add_layer(Dense::new(10, Activation::Sigmoid))
        .on_epoch_with_loss(move |epoch, total_epochs, train_loss, test_loss| {
            println!("[{:?}], epoch {} of {} -> train_loss: {}, test_loss: {}",
                now.elapsed(), epoch, total_epochs, train_loss, test_loss);
            mnist_epochs.push(epoch);
            mnist_train_loss.push(train_loss);
            mnist_test_loss.push(test_loss);
            fg.clear_axes();
            fg.axes2d()
                .set_title("MNIST - loss", &[])
                .set_legend(Graph(0.5), Graph(0.9), &[], &[])
                .set_x_label("epochs", &[])
                .set_y_label("loss", &[])
                .lines(mnist_epochs.iter(), mnist_train_loss.iter(), &[Caption("Train MSE")])
                .lines(mnist_epochs.iter(), mnist_test_loss.iter(), &[Caption("Test MSE")]);
            fg.show().unwrap();
            now = Instant::now();
        })
        .train(&train_x, &train_y, &test_x, &test_y, 3.0, 30, 100);
    println!("Total time: {:?}", total.elapsed());

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
