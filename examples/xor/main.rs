#[macro_use]
extern crate neuro;

use gnuplot::*;
use neuro::{Mtx, Neuro};
use neuro::layers::{Activation, Dense};

fn main() {

    let x = mtx! {
        (4, 2);
        [           // Y
            0, 0,   // 0
            0, 1,   // 1
            1, 0,   // 1
            1, 1    // 0
        ]
    };

    let y = mtx!{(4, 1); [0, 1, 1, 0]};

    // test dataset
    let test = x.clone();

    let mut xor_epochs: Vec<u64> = vec![];
    let mut xor_train_mse: Vec<f32> = vec![];
    let mut xor_test_mse: Vec<f32> = vec![];
    let mut fg = Figure::new();

    let mut xor = Neuro::new()
        .add_layer(Dense::new(3, Activation::Tanh))
        .add_layer(Dense::new(1, Activation::Tanh))
        .on_epoch(move |epoch, total_epochs, train_mse, test_mse| {
            if epoch % 10 != 0 {
                return;
            }
            xor_epochs.push(epoch);
            xor_train_mse.push(train_mse);
            xor_test_mse.push(test_mse);
            fg.clear_axes();
            fg.axes2d()
                .set_title("XOR - loss", &[])
                .set_legend(Graph(0.5), Graph(0.9), &[], &[])
                .set_x_label("epochs", &[])
                .set_y_label("loss", &[])
                .lines(xor_epochs.iter(), xor_train_mse.iter(), &[Caption("Train MSE")])
                .lines(xor_epochs.iter(), xor_test_mse.iter(), &[Caption("Test MSE")]);
            fg.show().unwrap();
        })
        .train(&x, &y, &test, &y, 0.1, 10000, 100);

    println!("{:?}", xor.predict(&test).unwrap().get_raw());
}
