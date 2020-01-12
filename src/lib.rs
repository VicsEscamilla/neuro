mod oclot;

extern crate rand;

use rand::seq::SliceRandom;
use std::f32::consts::E;
use rand::Rng;
pub use oclot::Mtx;

pub enum Activation {
    Sigmoid,
    Tanh,
    ReLU
}

pub enum Runtime {
    CPU,
    GPU
}

#[derive(Debug)]
pub enum NeuroError {
    ModelNotTrained
}

pub struct Neuro {
    layers: Vec<Layer>,
    weights: Vec<Mtx>,
    biases: Vec<Vec<f32>>,
    gpu: Option<oclot::Oclot>,
    on_epoch_fn: Option<Box<dyn FnMut(u64, u64, f32, f32)>>
}

struct Layer {
    neurons: u64,
    activation: Activation,
}

impl Neuro {
    pub fn new(runtime:Runtime) -> Self {
        Neuro {
            layers: vec![],
            weights: vec![],
            biases: vec![],
            gpu: match runtime {
                Runtime::CPU => None,
                Runtime::GPU => Some(oclot::Oclot::new())
            },
            on_epoch_fn: None
        }
    }

    pub fn add_layer(mut self, neurons:u64, activation:Activation) -> Self {
        if self.layers.is_empty() {
            self.layers = vec![Layer{neurons, activation}];
        } else {
            self.layers.push(Layer{neurons, activation});
        }
        self
    }

    pub fn train(mut self, x:&Mtx, y:&Mtx, test_x:&Mtx, test_y:&Mtx,
        learning_rate:f32, epochs:u64, batch_size:usize) -> Self {
        if self.layers.is_empty() {
            return self;
        }

        self.init_parameters(x.shape().1);
        for epoch in 0..epochs {
            let mut order: Vec<usize> = (0..x.shape().0).collect();
            order.shuffle(&mut rand::thread_rng());

            let epoch_x_raw = x.reorder_rows(&order).get_raw();
            let epoch_y_raw = y.reorder_rows(&order).get_raw();

            let (rows, x_cols) = x.shape();
            let (_, y_cols) = y.shape();

            let mut _batch_size = 1;
            if batch_size > 0 {
                _batch_size = batch_size;
            }

            let mut total_batches = rows/_batch_size;
            if rows % _batch_size != 0 {
                total_batches += 1;
            }

            for batch in 0..total_batches {
                let x_first = batch*_batch_size*x_cols;
                let mut x_last = x_first + _batch_size*x_cols;
                if x_last > rows*x_cols {
                    x_last = rows*x_cols;
                }

                let y_first = batch*_batch_size*y_cols;
                let mut y_last = y_first + _batch_size*y_cols;
                if y_last > rows*y_cols {
                    y_last = rows*y_cols;
                }

                let mini_x = Mtx::new(((x_last-x_first)/x_cols, x_cols), epoch_x_raw[x_first..x_last].to_vec());
                let mini_y = Mtx::new(((y_last-y_first)/y_cols, y_cols), epoch_y_raw[y_first..y_last].to_vec());
                let (_, activations) = self.feedforward(&mini_x);
                let (dw, db) = self.backpropagation(&activations, &mini_y);
                for i in 0..self.weights.len() {
                    self.weights[i] = self.weights[i].add(
                        &dw[i].func(|&x| x*(learning_rate/mini_x.shape().0 as f32)));
                    self.biases[i] = self.biases[i].iter()
                                       .zip(&db[i].func(|x| x*(learning_rate/mini_y.shape().0 as f32)).get_raw())
                                       .map(|(&a, &b)| a+b)
                                       .collect();
                }
            }

            let get_msr = &mut |x:&Mtx, y:&Mtx| {
                let prediction = self.predict(&x).unwrap();
                let (tests, classes) = y.shape();
                prediction.add(&y.func(|x|-x))
                    .func(|x|x*x)
                    .sum(0)
                    .func(|x|x/classes as f32)
                    .func(|x|x.sqrt())
                    .sum(1)
                    .func(|x|x/tests as f32)
                    .get_raw()[0]
            };

            // calculate loss
            let train_msr = get_msr(&x, &y);
            let test_msr = get_msr(&test_x, &test_y);
            if let Some(func) = &mut self.on_epoch_fn {
                func(epoch, epochs, train_msr, test_msr);
            }
        }
        self
    }

    pub fn predict(&mut self, x:&Mtx) -> Result<Mtx, NeuroError> {
        if self.weights.is_empty() {
            return Err(NeuroError::ModelNotTrained);
        }

        let (_, activations) = self.feedforward(x);
        Ok(activations[activations.len()-1].clone())
    }

    pub fn on_epoch<F:FnMut(u64, u64, f32, f32) + 'static>(mut self, func: F) -> Self {
        self.on_epoch_fn = Some(Box::new(func));
        self
    }


    fn init_parameters(&mut self, input_size: usize) {
        if self.layers.is_empty() {
            return;
        }

        let rows = input_size;
        let cols = self.layers[0].neurons as usize;
        let mut w: Vec<Mtx> = Vec::with_capacity(self.layers.len()-1);
        let mut b: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len()-1);
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

    fn feedforward(&mut self, x: &Mtx) -> (Vec<Mtx>, Vec<Mtx>) {
        let mut caches = Vec::with_capacity(self.layers.len());
        let mut activations = Vec::with_capacity(self.layers.len()+1);
        activations.push(x.clone());

        for i in 0..self.layers.len() {
            match &mut self.gpu {
                Some(gpu) => {
                    caches.push(gpu.dot(&activations[i], &self.weights[i])
                                .add_vector(&self.biases[i]));
                },
                None => {
                    caches.push(activations[i].dot(&self.weights[i])
                                .add_vector(&self.biases[i]));
                }
            };
            activations.push(match &self.layers[i].activation {
                Activation::Sigmoid => caches[i].func(Neuro::sigmoid),
                Activation::Tanh => caches[i].func(Neuro::tanh),
                Activation::ReLU => caches[i].func(Neuro::relu)
            });
        }
        (caches, activations)
    }

    fn backpropagation(&mut self, activations: &Vec<Mtx>, y:&Mtx) -> (Vec<Mtx>, Vec<Mtx>){
        let mut deriv_b: Vec<Mtx> = Vec::with_capacity(activations.len());
        let mut deriv_w: Vec<Mtx> = Vec::with_capacity(activations.len());

        let mut delta = activations[activations.len()-1].func(|&x|-x).add(&y)
            .prod(&match &self.layers[self.layers.len()-1].activation {
                    Activation::Sigmoid => activations[activations.len()-1].func(Neuro::sigmoid_prime),
                    Activation::Tanh => activations[activations.len()-1].func(Neuro::tanh_prime),
                    Activation::ReLU => activations[activations.len()-1].func(Neuro::relu_prime)
                });

        match &mut self.gpu {
            Some(gpu) => deriv_w.push(gpu.dot(&activations[activations.len()-2].trans(), &delta)),
            None => deriv_w.push(activations[activations.len()-2].trans().dot(&delta))
        };
        deriv_b.push(delta.sum(1));

        for i in (1..self.layers.len()).rev() {
            match &mut self.gpu {
                Some(gpu) => {
                    delta = gpu.dot(&delta, &self.weights[i].trans())
                        .prod(&match &self.layers[i].activation {
                            Activation::Sigmoid => activations[i].func(Neuro::sigmoid_prime),
                            Activation::Tanh => activations[i].func(Neuro::tanh_prime),
                            Activation::ReLU => activations[i].func(Neuro::relu_prime)
                        });
                    deriv_w.insert(0, gpu.dot(&activations[i-1].trans(), &delta));
                },
                None => {
                    delta = delta.dot(&self.weights[i].trans())
                        .prod(&match &self.layers[i].activation {
                            Activation::Sigmoid => activations[i].func(Neuro::sigmoid_prime),
                            Activation::Tanh => activations[i].func(Neuro::tanh_prime),
                            Activation::ReLU => activations[i].func(Neuro::relu_prime)
                        });
                    deriv_w.insert(0, activations[i-1].trans().dot(&delta));
                }
            };
            deriv_b.insert(0, delta.sum(1));
        }

        (deriv_w, deriv_b)
    }

    fn random_vector(size: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        vec![0.; size].iter().map(|_| rng.gen::<f32>()).collect()
    }

    fn sigmoid(x: &f32) -> f32 {
        1. / (1. + E.powf(-x))
    }

    fn sigmoid_prime(x: &f32) -> f32 {
        Neuro::sigmoid(x) * (1. - Neuro::sigmoid(x))
    }

    fn tanh(x: &f32) -> f32 {
        x.tanh()
    }

    fn tanh_prime(x: &f32) -> f32 {
        1. - x*x
    }

    fn relu(x: &f32) -> f32 {
        if *x > 0. {
            *x
        } else {
            0.
        }
    }

    fn relu_prime(x: &f32) -> f32 {
        if *x > 0. {
            1.
        } else {
            0.
        }
    }
}
