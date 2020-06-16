mod oclot;

use rand::Rng;
use super::{Mtx, mtx, Layer, activation::{Activation, function, prime}};

pub struct Dense {
    neurons: usize,
    activation: Activation,
    weights: Mtx,
    biases: Vec<f32>,
    dw: Mtx,
    db: Mtx,
    gpu: oclot::Oclot
}


impl Dense {
    pub fn new(neurons:usize, activation: Activation) -> Box<Dense> {
        Box::new(Dense {
            neurons,
            activation,
            weights: mtx![],
            biases: vec![],
            dw: mtx![],
            db: mtx![],
            gpu: oclot::Oclot::new()
        })
    }

    fn random_vector(size: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        vec![0.; size].iter().map(|_| rng.gen::<f32>()).collect()
    }
}


impl Layer for Dense {
    fn forward(&mut self, x: &Mtx) -> Mtx {
        self.gpu.forward(&x, &self.weights, &self.biases)
                .func(function(&self.activation))
    }


    fn backward(&mut self, x: &Mtx, delta:&Mtx) -> Mtx {
        self.dw = self.gpu.dot(&x.trans(), &delta);
        self.db = delta.sum(1);
        self.gpu.dot(&delta, &self.weights.trans())
             .prod(&x.func(prime(&self.activation)))
    }


    fn update(&mut self, rate: f32) {
        self.weights = self.weights.add(&self.dw.func(|&x| x*rate));
        self.biases = self.biases.iter()
             .zip(&self.db.func(|x| x*rate).get_raw())
             .map(|(&a, &b)| a+b)
             .collect();
    }


    fn initialize(&mut self, input_size: usize) {
        let (rows, cols) = (input_size, self.neurons);
        self.weights = Mtx::new((rows, cols), Dense::random_vector(rows*cols));
        self.biases = Dense::random_vector(cols);
    }


    fn input_size(&self) -> usize {
        self.neurons
    }


    fn error(&self, result: &Mtx, y: &Mtx) -> Mtx {
        // Cross-entropy?
        result.func(|&x|-x).add(&y)

        // Squared error?
        // result.func(|&x|-x).add(&y)
        //       .prod(&result.func(prime(&self.activation)))
    }
}
