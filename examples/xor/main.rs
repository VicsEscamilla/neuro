extern crate neuro;

extern crate npy;

use neuro::{Neuro, Activation, Mtx};

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

    let mut xor = Neuro::new()
        .add_layer(3, Activation::Tanh)
        .add_layer(1, Activation::Tanh)
        .train(&x, &y, 0.01, 10000, 100, 1000);

    println!("{:?}", xor.predict(&test).unwrap().get_raw());
}
