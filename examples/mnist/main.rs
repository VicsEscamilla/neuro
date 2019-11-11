extern crate neuro;

extern crate npy;

use neuro::{Neuro, Activation, Mtx};

use std::io::Read;
use npy::NpyData;

fn load_file(file: &str) -> Vec<f32> {
    let mut buf = vec![];
    std::fs::File::open(file).unwrap().read_to_end(&mut buf).unwrap();
    return NpyData::from_bytes(&buf).unwrap().to_vec();
}

fn main() {
    println!("LOADING train_x");
    let train_x_original: Vec<f32> = load_file("data/mnist_train_X.npy");
    println!("LOADING train_y");
    let train_y_original: Vec<f32> = load_file("data/mnist_train_y.npy");
    println!("LOADING test_x");
    let test_x_original: Vec<f32> = load_file("data/mnist_test_X.npy");
    println!("LOADING test_y");
    let test_y_original: Vec<f32> = load_file("data/mnist_test_y.npy");

    println!("train_X -> {}", train_x_original.len());
    println!("train_y -> {}", train_y_original.len());
    println!("test_X -> {}", test_x_original.len());
    println!("test_X -> {}", test_y_original.len());

    let train_x = Mtx::new((50000, 784), train_x_original);
    println!("Created train_x Mtx with shape {:?}", train_x.shape());
    let train_y = Mtx::new((50000, 10), train_y_original);
    println!("Created train_y Mtx with shape {:?}", train_y.shape());
    let test_x = Mtx::new((10000, 784), test_x_original);
    println!("Created test_x Mtx with shape {:?}", test_x.shape());
    let test_y = Mtx::new((10000, 10), test_y_original);
    println!("Created test_y Mtx with shape {:?}", test_y.shape());

    let mut digit = Neuro::new()
        .add_layer(30, Activation::Sigmoid)
        .add_layer(10, Activation::Sigmoid)
        .train(&train_x, &train_y, 3.0, 30, 100);

    for i in 0..test_x.shape().0 {
        let ypos = i*10;
        let xpos = i*784;
        let expected = &test_y.get_raw()[ypos..ypos+10];
        let input = Mtx::new((1, 784), test_x.get_raw()[xpos..xpos+784].to_vec());
        let got = digit.predict(&input).unwrap().get_raw();
        println!("Expected:  {:?}", expected);
        println!("Predicted: {:?}", got);
        println!();
        std::thread::sleep(std::time::Duration::from_secs(3));
    }

}
