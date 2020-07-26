extern crate ocl;
mod oclot;

use rand::Rng;
use rand::distributions::StandardNormal;
use super::{Mtx, mtx, Layer, activation::{Activation, function, prime}};

pub struct Dense {
    neurons: usize,
    activation: Activation,
    weights: Mtx,
    biases: Mtx,
    dw: Mtx,
    db: Mtx,
    gpu: oclot::Oclot
}


impl Dense {
    pub fn new(neurons:usize, activation: Activation) -> Box<Dense> {
        let gpu = oclot::Oclot::new();
        Box::new(Dense {
            neurons,
            activation,
            weights: mtx![],
            biases: mtx![],
            dw: mtx![],
            db: mtx![],
            gpu: oclot::Oclot::new()
        })
    }

    fn random_vector(size: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        rng.sample_iter(&StandardNormal)
           .take(size).map(|x| x as f32).collect()
    }
}


impl Layer for Dense {
    fn forward(&mut self, x: &Mtx) -> (Mtx, Mtx) {
        let c = self.gpu.forward(&x, &self.weights, &self.biases.get_raw());
        let a = c.func(function(&self.activation));
        (c, a)
    }


    fn backward(&mut self, c: &Mtx, a: &Mtx, delta:&Mtx) -> Mtx {
        let xtrans = self.gpu.trans(&a);
        self.dw = self.gpu.dot(&xtrans, &delta);
        self.db = self.gpu.sum_rows(&delta);
        let wtrans = self.gpu.trans(&self.weights);
        self.gpu.dot(&delta, &wtrans)
             .prod(&c.func(prime(&self.activation)))
    }


    fn update(&mut self, rate: f32) {
        self.weights = self.weights.add(&self.dw.func(|&x| -x*rate));
        self.biases = self.biases.add(&self.db.func(|&x| -x*rate));
    }


    fn initialize(&mut self, input_size: usize) {
        let (rows, cols) = (input_size, self.neurons);
        self.weights = Mtx::new((rows, cols), Dense::random_vector(rows*cols));
        self.biases = Mtx::new((1, cols), Dense::random_vector(cols));
    }


    fn input_size(&self) -> usize {
        self.neurons
    }


    fn error(&self, c:&Mtx, a:&Mtx, y:&Mtx) -> Mtx {
        a.add(&y.func(|&x|-x))
         .prod(&c.func(prime(&self.activation)))
    }

}
