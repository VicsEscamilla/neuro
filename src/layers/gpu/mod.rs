extern crate ocl;
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
    w_buf: ocl::Buffer<f32>,
    b_buf: ocl::Buffer<f32>,
    dw_buf: ocl::Buffer<f32>,
    db_buf: ocl::Buffer<f32>,
    fw_input_buf: ocl::Buffer<f32>,
    fw_output_buf: ocl::Buffer<f32>,
    bw_input_buf: ocl::Buffer<f32>,
    bw_input_trans_buf: ocl::Buffer<f32>,
    bw_delta_buf: ocl::Buffer<f32>,
    bw_delta_trans_buf: ocl::Buffer<f32>,
    bw_w_trans_buf: ocl::Buffer<f32>,
    bw_output_buf: ocl::Buffer<f32>,
    gpu: oclot::Oclot
}


impl Dense {
    pub fn new(neurons:usize, activation: Activation) -> Box<Dense> {
        let gpu = oclot::Oclot::new();
        Box::new(Dense {
            neurons,
            activation,
            weights: mtx![],
            biases: vec![],
            dw: mtx![],
            db: mtx![],
            w_buf: gpu.create_buffer(1),
            b_buf: gpu.create_buffer(1),
            dw_buf: gpu.create_buffer(1),
            db_buf: gpu.create_buffer(1),
            fw_input_buf: gpu.create_buffer(1),
            fw_output_buf: gpu.create_buffer(1),
            bw_input_buf: gpu.create_buffer(1),
            bw_input_trans_buf: gpu.create_buffer(1),
            bw_delta_buf: gpu.create_buffer(1),
            bw_delta_trans_buf: gpu.create_buffer(1),
            bw_w_trans_buf: gpu.create_buffer(1),
            bw_output_buf: gpu.create_buffer(1),
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
        let (m, n, k) = (x.shape().0, self.weights.shape().1, x.shape().1);

        if self.fw_input_buf.len() == 1 || self.fw_input_buf.len() < x.size() {
            self.fw_input_buf = self.gpu.create_buffer(x.size());
        }

        if self.fw_output_buf.len() == 1 || self.fw_output_buf.len() < m*n {
            self.fw_output_buf = self.gpu.create_buffer(m*n);
        }

        let mut result = vec![0.; m*n];
        self.gpu.write_buffer(&x.get_raw(), &mut self.fw_input_buf);
        self.gpu.forward_buf(m, n, k, &self.fw_input_buf, &self.w_buf,
                         &self.b_buf, &mut self.fw_output_buf);
        self.gpu.read_buffer(&self.fw_output_buf, &mut result);

        Mtx::new((m, n), result)
            .func(function(&self.activation))

        // Backup plan!
        // self.gpu.forward(&x, &self.weights, &self.biases)
        //         .func(function(&self.activation))
    }


    fn backward(&mut self, x: &Mtx, delta:&Mtx) -> Mtx {
        let (input_rows, input_cols) = x.shape();
        let (delta_rows, delta_cols) = delta.shape();
        let (weights_rows, weights_cols) = self.weights.shape();
        let biases_size = self.biases.len();
        let (m, n) = (delta_rows, weights_rows);

        if self.bw_input_buf.len() == 1 || self.bw_input_buf.len() < x.size() {
            self.bw_input_buf = self.gpu.create_buffer(x.size());
            self.bw_input_trans_buf = self.gpu.create_buffer(x.size());
        }

        if self.bw_delta_buf.len() == 1 || self.bw_delta_buf.len() < delta.size() {
            self.bw_delta_buf = self.gpu.create_buffer(delta.size());
            self.bw_delta_trans_buf = self.gpu.create_buffer(delta.size());
        }

        if self.bw_output_buf.len() == 1 || self.bw_output_buf.len() < m*n {
            self.bw_output_buf = self.gpu.create_buffer(m*n);
        }

        if self.bw_w_trans_buf.len() == 1 || self.bw_w_trans_buf.len() < self.weights.size() {
            self.bw_w_trans_buf = self.gpu.create_buffer(self.weights.size());
        }

        let mut output = vec![0.; m*n];
        self.gpu.write_buffer(&x.get_raw(), &mut self.bw_input_buf);
        self.gpu.write_buffer(&delta.get_raw(), &mut self.bw_delta_buf);
        self.gpu.backward_buf(input_rows, input_cols, delta_rows, delta_cols,
                              weights_rows, weights_cols, biases_size,
                              &self.bw_input_buf, &self.bw_delta_buf,
                              &self.w_buf, &self.b_buf,
                              &self.dw_buf, &self.db_buf,
                              &self.bw_input_trans_buf, &self.bw_delta_trans_buf,
                              &self.bw_w_trans_buf, &self.bw_output_buf);
        self.gpu.read_buffer(&self.bw_output_buf, &mut output);

        Mtx::new((m, n), output).prod(&x.func(prime(&self.activation)))

        // Backup plan!
        // let xtrans = self.gpu.trans(&x);
        // self.dw = self.gpu.dot(&xtrans, &delta);
        // self.db = self.gpu.sum_rows(&delta);
        // let wtrans = self.gpu.trans(&self.weights);
        // self.gpu.dot(&delta, &wtrans)
        //      .prod(&x.func(prime(&self.activation)))
    }


    fn update(&mut self, rate: f32) {
        let (rows, cols) = self.weights.shape();
        self.gpu.update_buf(rows, cols, rate, &self.dw_buf, &self.w_buf);

        let (rows, cols) = (1, self.biases.len());
        self.gpu.update_buf(rows, cols, rate, &self.db_buf, &self.b_buf);

        // Backup plan!
        // self.weights = self.weights.add(&self.dw.func(|&x| x*rate));
        // self.biases = self.biases.iter()
        //      .zip(&self.db.func(|x| x*rate).get_raw())
        //      .map(|(&a, &b)| a+b)
        //      .collect();
    }


    fn initialize(&mut self, input_size: usize) {
        let (rows, cols) = (input_size, self.neurons);
        self.weights = Mtx::new((rows, cols), Dense::random_vector(rows*cols));
        self.dw = mtx!{(rows, cols); 0.};
        self.biases = Dense::random_vector(cols);
        self.db = mtx![(1, cols); 0.];

        self.w_buf = self.gpu.create_buffer(rows*cols);
        self.dw_buf = self.gpu.create_buffer(rows*cols);
        self.gpu.write_buffer(&self.weights.get_raw(), &mut self.w_buf);

        self.b_buf = self.gpu.create_buffer(cols);
        self.db_buf = self.gpu.create_buffer(cols);
        self.gpu.write_buffer(&self.biases, &mut self.b_buf);
    }


    fn input_size(&self) -> usize {
        self.neurons
    }

}
