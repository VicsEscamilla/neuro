#[macro_use]
mod linalg;
pub mod layers;

extern crate rand;

use rand::seq::SliceRandom;

pub use linalg::Mtx;


pub trait Layer {
    fn forward(&mut self, x: &Mtx) -> Mtx;
    fn backward(&mut self, x: &Mtx, delta:&Mtx) -> Mtx;
    fn update(&mut self, rate: f32);
    fn initialize(&mut self, input_size: usize);
    fn input_size(&self) -> usize;
    fn error(&self, result: &Mtx, y: &Mtx) -> Mtx;
}


#[derive(Debug)]
pub enum NeuroError {
    ModelNotTrained
}


pub struct Neuro {
    layers: Vec<Box<dyn Layer>>,
    on_epoch_fn: Option<Box<dyn FnMut(u64, u64, f32, f32)>>,
    is_initialized: bool
}


impl Neuro {
    pub fn new() -> Self {
        Neuro {
            layers: vec![],
            on_epoch_fn: None,
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
                let rows = x_batch.len()/x_cols;
                let mini_x = Mtx::new((rows, x_cols), x_batch.to_vec());
                let mini_y = Mtx::new((rows, y_cols), y_batch.to_vec());
                let activations = self.feedforward(&mini_x);
                self.backpropagation(&activations, &mini_y);
                self.update_model(learning_rate/rows as f32);
            }

            if self.on_epoch_fn.is_some() {
                // calculate loss
                let train_loss = self.get_loss(&x, &y);
                let test_loss = self.get_loss(&test_x, &test_y);
                self.on_epoch_fn.as_mut().unwrap()(epoch, epochs, train_loss, test_loss);
            }
        }

        self
    }


    pub fn predict(&mut self, x:&Mtx) -> Result<Mtx, NeuroError> {
        if !self.is_initialized {
            return Err(NeuroError::ModelNotTrained);
        }

        let activations = self.feedforward(x);
        Ok(activations.last().unwrap().clone())
    }


    pub fn on_epoch<F:FnMut(u64, u64, f32, f32) + 'static>(mut self, func: F) -> Self {
        self.on_epoch_fn = Some(Box::new(func));
        self
    }


    fn get_loss(&mut self, x:&Mtx, y:&Mtx) -> f32 {
        let prediction = &self.predict(&x).unwrap();
        let (tests, classes) = y.shape();

        // Cross-entropy
        let first = y.prod(&prediction.func(|x| x.ln()));
        let second = y.func(|x|1.-x)
                      .prod(&prediction.func(|x| (1.-x).ln()));
        first.add(&second)
            .sum(0)
            .func(|x|x/classes as f32)
            .sum(1)
            .func(|x|x/tests as f32)
            .func(|x|-x)
            .get_raw()[0]

        // msr
        // prediction.add(&y.func(|x|-x))
        //     .func(|x|x*x)
        //     .sum(0)
        //     .func(|x|x/classes as f32)
        //     .func(|x|x.sqrt())
        //     .sum(1)
        //     .func(|x|x/tests as f32)
        //     .get_raw()[0]
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


    fn feedforward(&mut self, x: &Mtx) -> Vec<Mtx> {
        let mut activations = Vec::with_capacity(self.layers.len()+1);
        activations.push(x.clone());
        for i in 0..self.layers.len() {
            activations.push(self.layers[i].forward(&activations[i]));
        }
        activations
    }


    fn backpropagation(&mut self, activations: &Vec<Mtx>, y:&Mtx) {
        let last_layer = self.layers.last().unwrap();
        let result = activations.last().unwrap();
        let mut delta = last_layer.error(&result, y);
        for i in (0..self.layers.len()).rev() {
            delta = self.layers[i].backward(&activations[i], &delta);
        }
    }


    fn update_model(&mut self, rate:f32) {
        for layer in &mut self.layers {
            layer.update(rate);
        }
    }

}
