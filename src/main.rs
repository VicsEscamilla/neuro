mod neuro;

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

    let xor = Neuro::new()
        .add_layer(2, Activation::Tanh)
        .add_layer(1, Activation::Tanh)
        .train(&x, &y, 0.1, 10000);

    println!("NEURO {:#?}", xor);

    println!("0 xor 0 = {}",
        xor.predict(&Mtx::new((1, 2), vec![h, h])).unwrap().get_raw()[0]);
    println!("0 xor 1 = {}",
        xor.predict(&Mtx::new((1, 2), vec![h, l])).unwrap().get_raw()[0]);
    println!("1 xor 0 = {}",
        xor.predict(&Mtx::new((1, 2), vec![l, h])).unwrap().get_raw()[0]);
    println!("1 xor 1 = {}",
        xor.predict(&Mtx::new((1, 2), vec![l, l])).unwrap().get_raw()[0]);
}
