extern crate neuro;

use neuro::{Neuro, Activation, Mtx, Runtime};

fn main() {

    //  XOR:
    //       X    Y
    //      0 0 | 0
    //      0 1 | 1
    //      1 0 | 1
    //      1 1 | 0

    let h = 1.;
    let l = -1.;

    let x = Mtx::new((4, 2), vec![
        l, l,
        l, h,
        h, l,
        h, h
    ]);

    let y = Mtx::new((4, 1), vec![
        l,
        h,
        h,
        l
    ]);

    let test = Mtx::new((4, 2), vec![
        l, l,
        l, h,
        h, l,
        h, h
    ]);

    let mut xor = Neuro::new(Runtime::CPU)
        .add_layer(3, Activation::Tanh)
        .add_layer(1, Activation::Tanh)
        .on_epoch(|epoch, total_epochs, train_mse, test_mse| {
            if epoch % 1000 == 0 {
                println!("epoch {} of {} -> train_mse: {}, test_mse: {}",
                    epoch, total_epochs, train_mse, test_mse);
            }
        })
        .train(&x, &y, &test, &y, 0.01, 10000, 100);

    println!("{:?}", xor.predict(&test).unwrap().get_raw());
}
