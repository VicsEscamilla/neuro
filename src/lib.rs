mod oclot;

extern crate rand;

use std::io::Write;
use std::io::Read;
use serde::{Serialize, Deserialize};
use rand::seq::SliceRandom;
use std::f32::consts::E;
use rand::Rng;
pub use oclot::Mtx;



#[derive(Serialize, Deserialize, Clone)]
pub enum Activation {
    Sigmoid = 1,
    Tanh = 2,
    ReLU = 3
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


#[derive(Serialize, Deserialize)]
struct LightNeuro {
    layers: Vec<Layer>,
    weights: Vec<Mtx>,
    biases: Vec<Vec<f32>>
}


#[derive(Serialize, Deserialize, Clone)]
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


    pub fn save(model: &Neuro, filename: String) {
        let mut file = std::fs::File::create(filename).unwrap();
        file.write(serde_json::to_string(&LightNeuro{
            layers: model.layers.clone(),
            weights: model.weights.clone(),
            biases: model.biases.clone()
        }).unwrap().as_bytes()).unwrap();
    }


    pub fn load(filename: String) -> Self {
        let mut file = std::fs::File::open(filename).unwrap();
        let mut json = String::new();
        file.read_to_string(&mut json).unwrap();
        let light: LightNeuro = serde_json::from_str(&json).unwrap();
        Neuro {
            layers: light.layers.clone(),
            weights: light.weights.clone(),
            biases: light.biases.clone(),
            gpu: None,
            on_epoch_fn: None
        }
    }


    pub fn with_runtime(mut self, runtime:Runtime) -> Self {
        self.gpu = match runtime {
            Runtime::CPU => None,
            Runtime::GPU => Some(oclot::Oclot::new())
        };
        self
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

        let batch_size = if batch_size == 0 {1} else {batch_size};
        let ((_, x_cols), (_, y_cols)) = (x.shape(), y.shape());
        let mut order: Vec<usize> = (0..x.shape().0).collect();
        for epoch in 0..epochs {
            order.shuffle(&mut rand::thread_rng());

            let epoch_x = x.reorder_rows(&order).get_raw();
            let x_iter = epoch_x.chunks(x_cols*batch_size);

            let epoch_y = y.reorder_rows(&order).get_raw();
            let y_iter = epoch_y.chunks(y_cols*batch_size);

            for (x_batch, y_batch) in x_iter.zip(y_iter) {
                let rows = x_batch.len()/x_cols;
                let mini_x = Mtx::new((rows, x_cols), x_batch.to_vec());
                let mini_y = Mtx::new((rows, y_cols), y_batch.to_vec());
                let (_, activations) = self.feedforward(&mini_x);
                let (dw, db) = self.backpropagation(&activations, &mini_y);
                self.update_model(dw, db, learning_rate/rows as f32);
            }

            if self.on_epoch_fn.is_some() {
                // calculate loss
                let train_msr = self.get_msr(&x, &y);
                let test_msr = self.get_msr(&test_x, &test_y);
                self.on_epoch_fn.as_mut().unwrap()(epoch, epochs, train_msr, test_msr);
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


    fn get_msr(&mut self, x:&Mtx, y:&Mtx) -> f32 {
        let prediction = &self.predict(&x).unwrap();
        let (tests, classes) = y.shape();
        prediction.add(&y.func(|x|-x))
            .func(|x|x*x)
            .sum(0)
            .func(|x|x/classes as f32)
            .func(|x|x.sqrt())
            .sum(1)
            .func(|x|x/tests as f32)
            .get_raw()[0]
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


    fn update_model(&mut self, dw: Vec<Mtx>, db: Vec<Mtx>, rate:f32) {
        for i in 0..self.weights.len() {
            self.weights[i] = self.weights[i].add(&dw[i].func(|&x| x*rate));
            self.biases[i] = self.biases[i]
                                .iter()
                                .zip(&db[i].func(|x| x*rate).get_raw())
                                .map(|(&a, &b)| a+b)
                                .collect();
        }
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
