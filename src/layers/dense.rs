use rand::Rng;
use rand::distributions::StandardNormal;
use super::{Mtx, mtx, Layer, activation::{Activation, function, prime}};

#[derive(Clone)]
pub struct Dense {
    neurons: usize,
    activation: Activation,
    weights: Mtx,
    biases: Mtx,
    dw: Mtx,
    db: Mtx
}


impl Dense {
    pub fn new(neurons:usize, activation: Activation) -> Box<Dense> {
        Box::new(Dense {
            neurons,
            activation,
            weights: mtx![],
            biases: mtx![],
            dw: mtx![],
            db: mtx![]
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
        let c = x.dot(&self.weights)
                 .add_vector(&self.biases.get_raw());
        let a = c.func(function(&self.activation));
        (c, a)
    }


    fn backward(&mut self, c: &Mtx, a: &Mtx, delta:&Mtx) -> Mtx {
        self.dw = a.trans().dot(&delta);
        self.db = delta.sum_rows();
        delta.dot(&self.weights.trans())
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

}
