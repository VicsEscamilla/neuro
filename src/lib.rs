#[macro_use]
mod linalg;
pub mod layers;

extern crate rand;

use rand::seq::SliceRandom;
use std::time::Instant;

pub use linalg::Mtx;


pub trait Layer {
    fn forward(&mut self, x: &Mtx) -> (Mtx, Mtx);
    fn backward(&mut self, c: &Mtx, a: &Mtx, delta:&Mtx) -> Mtx;
    fn update(&mut self, rate: f32);
    fn initialize(&mut self, input_size: usize);
    fn input_size(&self) -> usize;
    fn error(&self, c:&Mtx, a:&Mtx, y:&Mtx) -> Mtx;
}


#[derive(Debug)]
pub enum NeuroError {
    ModelNotTrained
}


pub struct Neuro {
    layers: Vec<Box<dyn Layer>>,
    on_epoch_fn: Option<Box<dyn FnMut(u64, u64)>>,
    on_epoch_with_loss_fn: Option<Box<dyn FnMut(u64, u64, f32, f32)>>,
    is_initialized: bool
}


impl Neuro {
    pub fn new() -> Self {
        Neuro {
            layers: vec![],
            on_epoch_fn: None,
            on_epoch_with_loss_fn: None,
            is_initialized: false
        }
    }


    pub fn add_layer(mut self, layer: Box<dyn Layer>) -> Self {
        if self.layers.is_empty() {
            self.layers = vec![layer];
        } else {
            self.layers.push(layer);
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
                if x_batch.len() % x_cols != 0 {
                    println!("WARNING: batch_size is not multiple of input size");
                }
                let rows = x_batch.len()/x_cols;
                let mini_x = Mtx::new((rows, x_cols), x_batch.to_vec());
                let mini_y = Mtx::new((rows, y_cols), y_batch.to_vec());
                let (caches, activations) = self.feedforward(&mini_x);
                self.backpropagation(&caches, &activations, &mini_y);
                self.update_model(learning_rate/rows as f32);
            }

            if self.on_epoch_fn.is_some() {
                self.on_epoch_fn.as_mut().unwrap()(epoch, epochs);
            }

            if self.on_epoch_with_loss_fn.is_some() {
                // calculate loss
                let train_loss = self.get_loss(&x, &y);
                let test_loss = self.get_loss(&test_x, &test_y);
                self.on_epoch_with_loss_fn.as_mut().unwrap()(epoch, epochs, train_loss, test_loss);
            }
        }

        self
    }


    pub fn predict(&mut self, x:&Mtx) -> Result<Mtx, NeuroError> {
        if !self.is_initialized {
            return Err(NeuroError::ModelNotTrained);
        }

        let (_, activations) = self.feedforward(x);
        Ok(activations.last().unwrap().clone())
    }


    pub fn on_epoch_with_loss<F:FnMut(u64, u64, f32, f32) + 'static>(mut self, func: F) -> Self {
        self.on_epoch_with_loss_fn = Some(Box::new(func));
        self
    }


    pub fn on_epoch<F:FnMut(u64, u64) + 'static>(mut self, func: F) -> Self {
        self.on_epoch_fn = Some(Box::new(func));
        self
    }


    fn get_loss(&mut self, x:&Mtx, y:&Mtx) -> f32 {
        let prediction = &self.predict(&x).unwrap();
        let (tests, _) = y.shape();
        // y.add(&prediction.func(|x|-x))
		  // // Frobenius norm
		  // .func(|x|x*x)
		  // .sum_cols()
		  // .sum_rows()
		  // .func(|x|x.sqrt())
		  // // end of Frobenius norm
		  // .func(|x|0.5*x*x)
		  // .func(|x|x/tests as f32)
		  // .get_raw()[0]

        y.add(&prediction.func(|x|-x))
         .func(|x|x*x)
         .sum_cols()
         .func(|x|x/2.0 as f32)
         .sum_rows()
         .func(|x|x/tests as f32)
         .get_raw()[0]
    }


    fn init_parameters(&mut self, input_size: usize) {
        if self.layers.is_empty() {
            return;
        }

        self.is_initialized = true;
        let mut input_size = input_size;
        for layer in &mut self.layers {
            layer.initialize(input_size);
            input_size = layer.input_size();
        }
    }


    fn feedforward(&mut self, x: &Mtx) -> (Vec<Mtx>, Vec<Mtx>) {
        let mut activations = Vec::with_capacity(self.layers.len()+1);
        let mut caches = Vec::with_capacity(self.layers.len()+1);
        caches.push(x.clone());
        activations.push(x.clone());
        for i in 0..self.layers.len() {
            let (c, a) = self.layers[i].forward(&activations[i]);
            caches.push(c);
            activations.push(a);
        }
        (caches, activations)
    }


    fn backpropagation(&mut self, caches: &Vec<Mtx>, activations: &Vec<Mtx>, y:&Mtx) {
        let a = activations.last().unwrap();
        let c = caches.last().unwrap();
        let mut delta = self.layers.last().unwrap().error(&c, &a, y);
        for i in (0..self.layers.len()).rev() {
            delta = self.layers[i].backward(&caches[i], &activations[i], &delta);
        }
    }


    fn update_model(&mut self, rate:f32) {
        for layer in &mut self.layers {
            layer.update(rate);
        }
    }

}
