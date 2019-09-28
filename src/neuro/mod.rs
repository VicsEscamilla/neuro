mod linalg;
extern crate rand;

use std::f64::consts::E;
use rand::Rng;
pub use linalg::Mtx;

#[derive(Debug)]
pub enum Activation {
    Sigmoid,
    Tanh,
    ReLU
}

#[derive(Debug)]
pub enum NeuroError {
    ModelNotTrained
}

#[derive(Debug)]
pub struct Neuro {
    layers: Vec<Layer>,
    weights: Vec<Mtx>,
    biases: Vec<Vec<f64>>
}

#[derive(Debug)]
struct Layer {
    neurons: u64,
    activation: Activation,
}

impl Neuro {
    pub fn new() -> Neuro {
        Neuro {
            layers: vec![],
            weights: vec![],
            biases: vec![]
        }
    }

    pub fn add_layer(mut self, neurons:u64, activation:Activation) -> Neuro {
        if self.layers.is_empty() {
            self.layers = vec![Layer{neurons, activation}];
        } else {
            self.layers.push(Layer{neurons, activation});
        }
        self
    }

    pub fn train(mut self, x:&Mtx, y:&Mtx, learning_rate:f64, epochs:u64) -> Neuro {
        if self.layers.is_empty() {
            return self;
        }

        self.init_parameters(x.shape().1);
        for _ in 0..epochs {
            let (_, activations) = self.feedforward(x);
            let (dw, db) = self.backpropagation(&activations, &y);
            for i in 0..self.weights.len() {
                self.weights[i] = self.weights[i].add(&dw[i].func(|&x| x*learning_rate));
                self.biases[i] = self.biases[i].iter()
                                   .zip(&db[i].func(|x| (*x)*learning_rate).get_raw())
                                   .map(|(&a, &b)| a+b)
                                   .collect();
            }
        }
        self
    }

    pub fn predict(&self, x:&Mtx) -> Result<Mtx, NeuroError> {
        if self.weights.is_empty() {
            return Err(NeuroError::ModelNotTrained);
        }

        let (_, activations) = self.feedforward(x);
        // Ok(activations[activations.len()-1].func(|x| {
        //         if *x >= 0.5 {
        //             1.
        //         } else {
        //             0.
        //         }
        // }))
        Ok(activations[activations.len()-1].clone())
    }

    fn init_parameters(&mut self, input_size: usize) {
        if self.layers.is_empty() {
            return;
        }

        let rows = input_size;
        let cols = self.layers[0].neurons as usize;
        let mut w: Vec<Mtx> = Vec::with_capacity(self.layers.len()-1);
        let mut b: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len()-1);
        w.push(Mtx::new((rows, cols), Neuro::random_vector(rows*cols)));
        b.push(Neuro::random_vector(cols));
        for i in 1..self.layers.len() {
            let rows = self.layers[i-1].neurons as usize;
            let cols = self.layers[i].neurons as usize;
            w.push(Mtx::new((rows, cols), Neuro::random_vector(rows*cols)));
            b.push(Neuro::random_vector(cols));
        }
        self.weights = w;
        self.biases = b;
    }

    fn feedforward(&self, x: &Mtx) -> (Vec<Mtx>, Vec<Mtx>) {
        let mut caches = Vec::with_capacity(self.layers.len());
        let mut activations = Vec::with_capacity(self.layers.len()+1);
        activations.push(x.clone());

        for i in 0..self.layers.len() {
            caches.push(activations[i]
                        .dot(&self.weights[i])
                        .add_vector(&self.biases[i]));
            activations.push(match &self.layers[i].activation {
                Activation::Sigmoid => caches[i].func(Neuro::sigmoid),
                Activation::Tanh => caches[i].func(Neuro::tanh),
                Activation::ReLU => caches[i].func(Neuro::relu)
            });
        }
        (caches, activations)
    }

    fn backpropagation(&self, activations: &Vec<Mtx>, y:&Mtx) -> (Vec<Mtx>, Vec<Mtx>){
        let mut deriv_b: Vec<Mtx> = Vec::with_capacity(activations.len());
        let mut deriv_w: Vec<Mtx> = Vec::with_capacity(activations.len());

        let mut delta = y.add(&activations[activations.len()-1].func(|&x|-x))
            .prod(&match &self.layers[self.layers.len()-1].activation {
                    Activation::Sigmoid => activations[activations.len()-1].func(Neuro::sigmoid_prime),
                    Activation::Tanh => activations[activations.len()-1].func(Neuro::tanh_prime),
                    Activation::ReLU => activations[activations.len()-1].func(Neuro::relu_prime)
                });

        deriv_w.push(activations[activations.len()-2].trans().dot(&delta));
        deriv_b.push(delta.sum(1));

        for i in (1..self.layers.len()).rev() {
            delta = delta.dot(&self.weights[i].trans())
                .prod(&match &self.layers[i].activation {
                    Activation::Sigmoid => activations[i].func(Neuro::sigmoid_prime),
                    Activation::Tanh => activations[i].func(Neuro::tanh_prime),
                    Activation::ReLU => activations[i].func(Neuro::relu_prime)
                });
            deriv_w.insert(0, activations[i-1].trans().dot(&delta));
            deriv_b.insert(0, delta.sum(1));
        }

        (deriv_w, deriv_b)
    }

    fn random_vector(size: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        vec![0.; size].iter().map(|_| rng.gen::<f64>()).collect()
    }

    fn sigmoid(x: &f64) -> f64 {
        1. / (1. + E.powf(-x))
    }

    fn sigmoid_prime(x: &f64) -> f64 {
        x * (1. - x)
    }

    fn tanh(x: &f64) -> f64 {
        x.tanh()
    }

    fn tanh_prime(x: &f64) -> f64 {
        1. - x*x
    }

    fn relu(x: &f64) -> f64 {
        if *x > 0. {
            *x
        } else {
            0.
        }
    }

    fn relu_prime(x: &f64) -> f64 {
        if *x > 0. {
            1.
        } else {
            0.
        }
    }
}
