extern crate ocl;

use super::Mtx;
use std::time::Instant;

#[derive(Debug)]
pub struct Oclot {
    pq: ocl::ProQue,
    trans_kern: ocl::Kernel,
    sum_kern: ocl::Kernel,
    dot_kern: ocl::Kernel,
    forward_kern: ocl::Kernel,
    backward_kern: ocl::Kernel,
    add_kern: ocl::Kernel,
    product_kern: ocl::Kernel,
    update_kern: ocl::Kernel,
}


impl Oclot {
    pub fn new() -> Oclot {
        let (pq, trans_kern, sum_kern, dot_kern, forward_kern, backward_kern,
             add_kern, product_kern, update_kern) = Oclot::init_kernels();
        Oclot {
            pq,
            trans_kern,
            sum_kern,
            dot_kern,
            forward_kern,
            backward_kern,
            add_kern,
            product_kern,
            update_kern
        }
    }


    pub fn sum_rows(&mut self, x:&Mtx) -> Mtx {
        let xx = self.trans(&x);
        let (rows, cols) = xx.shape();
        Mtx::new(
            (1, rows),
            self._sum(1, rows, cols, &xx.get_raw())[0..rows].to_vec())
    }


    pub fn sum_cols(&mut self, x:&Mtx) -> Mtx {
        let (rows, cols) = x.shape();
        Mtx::new(
            (rows, 1),
            self._sum(0, rows, cols, &x.get_raw())[0..rows].to_vec())

    }


    pub fn trans(&mut self, x:&Mtx) -> Mtx {
        let (rows, cols) = x.shape();
        Mtx::new(
            (cols, rows),
            self._transpose(rows, cols, &x.get_raw()))
    }


    pub fn dot(&mut self, a:&Mtx, b:&Mtx) -> Mtx {
        Mtx::new(
            (a.shape().0, b.shape().1),
            self._dot(a.shape().0, b.shape().1, a.shape().1,
                      &a.get_raw(), &b.get_raw())
        )
    }


    pub fn forward(&mut self, x:&Mtx, w:&Mtx, b:&Vec<f32>) -> Mtx {
        Mtx::new(
            (x.shape().0, w.shape().1),
            self._forward(x.shape().0, w.shape().1, x.shape().1,
                      &x.get_raw(), &w.get_raw(), &b)
        )
    }


    pub fn create_buffer(&self, size: usize) -> ocl::Buffer<f32> {
        ocl::Buffer::builder()
            .queue(self.pq.queue().clone())
            .flags(ocl::MemFlags::new().read_write())
            .len(size)
            .build().unwrap()
    }


    pub fn write_buffer(&self, src:&Vec<f32>, dst:&ocl::Buffer<f32>) {
        unsafe {
            let mut map = dst.map().write().enq().unwrap();
            std::ptr::copy_nonoverlapping(src.as_ptr(), map.as_mut_ptr(), (*src).len());
            map.unmap().enq().unwrap();
        }
        self.pq.finish().unwrap();
    }


    pub fn read_buffer(&self, src:&ocl::Buffer<f32>, dst:&mut Vec<f32>) {
        unsafe {
            let mut map = src.map().read().enq().unwrap();
            std::ptr::copy_nonoverlapping(map.as_ptr(), dst.as_mut_ptr(), (*dst).len());
            map.unmap().enq().unwrap();
        }
        self.pq.finish().unwrap();
    }


    fn init_kernels() -> (ocl::ProQue, ocl::Kernel, ocl::Kernel, ocl::Kernel,
                          ocl::Kernel, ocl::Kernel, ocl::Kernel, ocl::Kernel, ocl::Kernel) {
        let file = format!("{}/{}", env!("CARGO_MANIFEST_DIR"),"src/layers/gpu/kernel.cl");
        let mut pb = ocl::Program::builder();
        pb.src_file(file);

        let pq = ocl::ProQue::builder()
                    .prog_bldr(pb)
                    .build().unwrap();

        let trans_kern = pq.kernel_builder("transpose")
            .arg_named("rows", 1 as u32)
            .arg_named("cols", 1 as u32)
            .arg_named("X", None::<&ocl::Buffer<f32>>)
            .arg_named("C", None::<&ocl::Buffer<f32>>)
            .build().unwrap();

        let sum_kern = pq.kernel_builder("sum")
            .arg_named("dim", 1 as u32)
            .arg_named("rows", 1 as u32)
            .arg_named("cols", 1 as u32)
            .arg_named("X", None::<&ocl::Buffer<f32>>)
            .arg_named("C", None::<&ocl::Buffer<f32>>)
            .build().unwrap();

        let dot_kern = pq.kernel_builder("dot")
            .arg_named("M", 1 as u32)
            .arg_named("N", 1 as u32)
            .arg_named("K", 1 as u32)
            .arg_named("A", None::<&ocl::Buffer<f32>>)
            .arg_named("B", None::<&ocl::Buffer<f32>>)
            .arg_named("C", None::<&ocl::Buffer<f32>>)
            .build().unwrap();

        let fw_kern = pq.kernel_builder("forward")
            .arg_named("M", 1 as u32)
            .arg_named("N", 1 as u32)
            .arg_named("K", 1 as u32)
            .arg_named("X", None::<&ocl::Buffer<f32>>)
            .arg_named("W", None::<&ocl::Buffer<f32>>)
            .arg_named("B", None::<&ocl::Buffer<f32>>)
            .arg_named("R", None::<&ocl::Buffer<f32>>)
            .build().unwrap();

        let bw_kern = pq.kernel_builder("backward")
            .arg_named("input_rows", 1 as u32)
            .arg_named("input_cols", 1 as u32)
            .arg_named("delta_rows", 1 as u32)
            .arg_named("delta_cols", 1 as u32)
            .arg_named("weights_rows", 1 as u32)
            .arg_named("weights_cols", 1 as u32)
            .arg_named("biases_size", 1 as u32)
            .arg_named("input", None::<&ocl::Buffer<f32>>)
            .arg_named("delta", None::<&ocl::Buffer<f32>>)
            .arg_named("weights", None::<&ocl::Buffer<f32>>)
            .arg_named("biases", None::<&ocl::Buffer<f32>>)
            .arg_named("d_weights", None::<&ocl::Buffer<f32>>)
            .arg_named("d_biases", None::<&ocl::Buffer<f32>>)
            .arg_named("input_trans", None::<&ocl::Buffer<f32>>)
            .arg_named("delta_trans", None::<&ocl::Buffer<f32>>)
            .arg_named("weights_trans", None::<&ocl::Buffer<f32>>)
            .arg_named("output", None::<&ocl::Buffer<f32>>)
            .build().unwrap();

        let add_kern = pq.kernel_builder("add_a_to_b")
            .arg_named("rows", 1 as u32)
            .arg_named("cols", 1 as u32)
            .arg_named("A", None::<&ocl::Buffer<f32>>)
            .arg_named("B", None::<&ocl::Buffer<f32>>)
            .build().unwrap();

        let product_kern = pq.kernel_builder("product")
            .arg_named("rows", 1 as u32)
            .arg_named("cols", 1 as u32)
            .arg_named("A", None::<&ocl::Buffer<f32>>)
            .arg_named("B", None::<&ocl::Buffer<f32>>)
            .arg_named("C", None::<&ocl::Buffer<f32>>)
            .build().unwrap();

        let update_kern = pq.kernel_builder("update")
            .arg_named("rows", 1 as u32)
            .arg_named("cols", 1 as u32)
            .arg_named("rate", 1.0 as f32)
            .arg_named("delta", None::<&ocl::Buffer<f32>>)
            .arg_named("input", None::<&ocl::Buffer<f32>>)
            .build().unwrap();

        (pq, trans_kern, sum_kern, dot_kern, fw_kern,
         bw_kern, add_kern, product_kern, update_kern)
    }


    fn _dot(&mut self, m:usize, n:usize, k:usize, a:&Vec<f32>, b:&Vec<f32>) -> Vec<f32> {
        self.pq.set_dims([m, n]);
        let mut response: Vec<f32> = vec![0.0f32; m*n];

        // let now = Instant::now();
        let device_a = unsafe {
            ocl::Buffer::builder()
                .queue(self.pq.queue().clone())
                .flags(ocl::MemFlags::new().read_only())
                .len(m*k)
                .use_host_slice(&a)
                .build().unwrap()
        };
        // println!("DEVICE A {:?}", now.elapsed());

        // let now = Instant::now();
        let device_b = unsafe {
            ocl::Buffer::builder()
                .queue(self.pq.queue().clone())
                .flags(ocl::MemFlags::new().read_only())
                .len(n*k)
                .use_host_slice(&b)
                .build().unwrap()
        };
        // println!("DEVICE B {:?}", now.elapsed());

        // let now = Instant::now();
        let device_c = unsafe {
            ocl::Buffer::builder()
                .queue(self.pq.queue().clone())
                .flags(ocl::MemFlags::new().read_write())
                .len(m*n)
                .use_host_slice(&response)
                .build().unwrap()
        };
        // println!("DEVICE C {:?}", now.elapsed());

        // let now = Instant::now();
        self.dot_kern.set_default_global_work_size(*self.pq.dims());
        self.dot_kern.set_arg("M", m as u32).unwrap();
        self.dot_kern.set_arg("N", n as u32).unwrap();
        self.dot_kern.set_arg("K", k as u32).unwrap();
        self.dot_kern.set_arg("A", Some::<&ocl::Buffer<f32>>(&device_a)).unwrap();
        self.dot_kern.set_arg("B", Some::<&ocl::Buffer<f32>>(&device_b)).unwrap();
        self.dot_kern.set_arg("C", Some::<&ocl::Buffer<f32>>(&device_c)).unwrap();

        unsafe {self.dot_kern.enq().unwrap();}
        // println!("KERN ENQUEUE {:?}", now.elapsed());

        // let now = Instant::now();
        // unsafe {
        //     let mut map_c = device_c.map().read().enq().unwrap();
        //     self.pq.finish().unwrap();

        //     std::ptr::copy_nonoverlapping(map_c.as_ptr(), response.as_mut_ptr(), m*n);
        //     map_c.unmap().enq().unwrap();
        // }
        device_c.read(&mut response).enq().unwrap();
        // println!("READ RESPONSE {:?}\n", now.elapsed());
        self.pq.finish().unwrap();

        return response;
    }


    fn _transpose(&mut self, rows:usize, cols:usize, x:&Vec<f32>) -> Vec<f32> {
        self.pq.set_dims([rows, cols]);
        let mut response: Vec<f32> = vec![0.0f32; rows*cols];

        let device_x = ocl::Buffer::builder()
            .queue(self.pq.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(rows*cols)
            .copy_host_slice(&x)
            .build().unwrap();

        let device_c = ocl::Buffer::builder()
            .queue(self.pq.queue().clone())
            .flags(ocl::MemFlags::new().read_write())
            .len(rows*cols)
            .build().unwrap();

        self.trans_kern.set_default_global_work_size(*self.pq.dims());
        self.trans_kern.set_arg("rows", rows as u32).unwrap();
        self.trans_kern.set_arg("cols", cols as u32).unwrap();
        self.trans_kern.set_arg("X", Some::<&ocl::Buffer<f32>>(&device_x)).unwrap();
        self.trans_kern.set_arg("C", Some::<&ocl::Buffer<f32>>(&device_c)).unwrap();

        unsafe {self.trans_kern.enq().unwrap();}
        device_c.read(&mut response).enq().unwrap();
        self.pq.finish().unwrap();

        return response;
    }


    fn _sum(&mut self, dim:usize, rows:usize, cols:usize, x:&Vec<f32>) -> Vec<f32> {
        self.pq.set_dims([rows, cols]);
        let mut response: Vec<f32> = vec![0.0f32; rows*cols];

        let device_x = ocl::Buffer::builder()
            .queue(self.pq.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(rows*cols)
            .copy_host_slice(&x)
            .build().unwrap();

        let device_c = ocl::Buffer::builder()
            .queue(self.pq.queue().clone())
            .flags(ocl::MemFlags::new().read_write())
            .len(rows*cols)
            .build().unwrap();

        self.sum_kern.set_default_global_work_size(*self.pq.dims());
        self.sum_kern.set_arg("dim", dim as u32).unwrap();
        self.sum_kern.set_arg("rows", rows as u32).unwrap();
        self.sum_kern.set_arg("cols", cols as u32).unwrap();
        self.sum_kern.set_arg("X", Some::<&ocl::Buffer<f32>>(&device_x)).unwrap();
        self.sum_kern.set_arg("C", Some::<&ocl::Buffer<f32>>(&device_c)).unwrap();

        unsafe {self.sum_kern.enq().unwrap();}
        device_c.read(&mut response).enq().unwrap();
        self.pq.finish().unwrap();

        return response;
    }


    fn _forward(&mut self, m:usize, n:usize, k:usize, x:&Vec<f32>, w:&Vec<f32>, b:&Vec<f32>) -> Vec<f32> {
        self.pq.set_dims([m, n]);
        let mut response: Vec<f32> = vec![0.0f32; m*n];

        let device_x = ocl::Buffer::builder()
            .queue(self.pq.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(m*k)
            .copy_host_slice(&x)
            .build().unwrap();

        let device_w = ocl::Buffer::builder()
            .queue(self.pq.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(n*k)
            .copy_host_slice(&w)
            .build().unwrap();

        let device_b = ocl::Buffer::builder()
            .queue(self.pq.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(n)
            .copy_host_slice(&b)
            .build().unwrap();

        let device_r = ocl::Buffer::builder()
            .queue(self.pq.queue().clone())
            .flags(ocl::MemFlags::new().read_write())
            .len(m*n)
            .build().unwrap();

        self.forward_kern.set_default_global_work_size(*self.pq.dims());
        self.forward_kern.set_arg("M", m as u32).unwrap();
        self.forward_kern.set_arg("N", n as u32).unwrap();
        self.forward_kern.set_arg("K", k as u32).unwrap();
        self.forward_kern.set_arg("X", Some::<&ocl::Buffer<f32>>(&device_x)).unwrap();
        self.forward_kern.set_arg("W", Some::<&ocl::Buffer<f32>>(&device_w)).unwrap();
        self.forward_kern.set_arg("B", Some::<&ocl::Buffer<f32>>(&device_b)).unwrap();
        self.forward_kern.set_arg("R", Some::<&ocl::Buffer<f32>>(&device_r)).unwrap();

        unsafe {self.forward_kern.enq().unwrap();}
        device_r.read(&mut response).enq().unwrap();
        self.pq.finish().unwrap();

        return response;
    }


    pub fn dot_buf(&mut self, m:usize, n:usize, k:usize, a:&ocl::Buffer<f32>,
               b:&ocl::Buffer<f32>, result:&ocl::Buffer<f32>) {

        self.pq.set_dims([m, n]);
        self.dot_kern.set_default_global_work_size(*self.pq.dims());
        self.dot_kern.set_arg("M", m as u32).unwrap();
        self.dot_kern.set_arg("N", n as u32).unwrap();
        self.dot_kern.set_arg("K", k as u32).unwrap();
        self.dot_kern.set_arg("A", Some::<&ocl::Buffer<f32>>(&a)).unwrap();
        self.dot_kern.set_arg("B", Some::<&ocl::Buffer<f32>>(&b)).unwrap();
        self.dot_kern.set_arg("C", Some::<&ocl::Buffer<f32>>(&result)).unwrap();

        unsafe {self.dot_kern.enq().unwrap();}
        self.pq.finish().unwrap();
    }


    pub fn forward_buf(&mut self, m:usize, n:usize, k:usize, x:&ocl::Buffer<f32>,
                   w:&ocl::Buffer<f32>, b:&ocl::Buffer<f32>, result:&ocl::Buffer<f32>) {
        self.pq.set_dims([m, n]);
        self.forward_kern.set_default_global_work_size(*self.pq.dims());
        self.forward_kern.set_arg("M", m as u32).unwrap();
        self.forward_kern.set_arg("N", n as u32).unwrap();
        self.forward_kern.set_arg("K", k as u32).unwrap();
        self.forward_kern.set_arg("X", Some::<&ocl::Buffer<f32>>(&x)).unwrap();
        self.forward_kern.set_arg("W", Some::<&ocl::Buffer<f32>>(&w)).unwrap();
        self.forward_kern.set_arg("B", Some::<&ocl::Buffer<f32>>(&b)).unwrap();
        self.forward_kern.set_arg("R", Some::<&ocl::Buffer<f32>>(&result)).unwrap();
        unsafe {self.forward_kern.enq().unwrap();}
        self.pq.finish().unwrap();
    }


    pub fn sum_buf(&mut self, dim:usize, rows:usize, cols:usize,
               x:&ocl::Buffer<f32>, result:&ocl::Buffer<f32>){
        self.pq.set_dims([rows, cols]);
        self.sum_kern.set_default_global_work_size(*self.pq.dims());
        self.sum_kern.set_arg("dim", dim as u32).unwrap();
        self.sum_kern.set_arg("rows", rows as u32).unwrap();
        self.sum_kern.set_arg("cols", cols as u32).unwrap();
        self.sum_kern.set_arg("X", Some::<&ocl::Buffer<f32>>(&x)).unwrap();
        self.sum_kern.set_arg("C", Some::<&ocl::Buffer<f32>>(&result)).unwrap();
        unsafe {self.sum_kern.enq().unwrap();}
        self.pq.finish().unwrap();
    }


    pub fn transpose_buf(&mut self, rows:usize, cols:usize, x:&ocl::Buffer<f32>,
                     result:&ocl::Buffer<f32>) {
        self.pq.set_dims([rows, cols]);
        self.trans_kern.set_default_global_work_size(*self.pq.dims());
        self.trans_kern.set_arg("rows", rows as u32).unwrap();
        self.trans_kern.set_arg("cols", cols as u32).unwrap();
        self.trans_kern.set_arg("X", Some::<&ocl::Buffer<f32>>(&x)).unwrap();
        self.trans_kern.set_arg("C", Some::<&ocl::Buffer<f32>>(&result)).unwrap();
        unsafe {self.trans_kern.enq().unwrap();}
        self.pq.finish().unwrap();
    }


    pub fn backward_buf(&mut self, input_rows:usize, input_cols:usize,
                        delta_rows:usize, delta_cols:usize,
                        weights_rows:usize, weights_cols:usize, biases_size:usize,
                        input:&ocl::Buffer<f32>, delta:&ocl::Buffer<f32>,
                        weights:&ocl::Buffer<f32>, biases:&ocl::Buffer<f32>,
                        d_weights:&ocl::Buffer<f32>, d_biases:&ocl::Buffer<f32>,
                        input_trans:&ocl::Buffer<f32>, delta_trans:&ocl::Buffer<f32>,
                        weights_trans:&ocl::Buffer<f32>, output:&ocl::Buffer<f32>) {

        self.transpose_buf(input_rows, input_cols, input, input_trans);
        self.transpose_buf(weights_rows, weights_cols, weights, weights_trans);
        self.transpose_buf(delta_rows, delta_cols, delta, delta_trans);

        // calculate d_biases
        self.sum_buf(0, delta_rows, delta_cols, delta, d_biases);

        // calculate d_weights
        // cols and rows from input are backward 'cause we're operating on the transpose
        self.dot_buf(input_cols, delta_cols, input_rows, input_trans, delta, d_weights);

        // calculate output
        // cols and rows from weights are backward 'cause we're operating on the transpose
        self.dot_buf(delta_rows, weights_rows, delta_cols, delta, weights_trans, output);
    }


    pub fn add_buf(&mut self, rows:usize, cols:usize,
                   a:&ocl::Buffer<f32>, b:&ocl::Buffer<f32>) {
        self.pq.set_dims([rows, cols]);
        self.add_kern.set_default_global_work_size(*self.pq.dims());
        self.add_kern.set_arg("rows", rows as u32).unwrap();
        self.add_kern.set_arg("cols", cols as u32).unwrap();
        self.add_kern.set_arg("A", Some::<&ocl::Buffer<f32>>(&a)).unwrap();
        self.add_kern.set_arg("B", Some::<&ocl::Buffer<f32>>(&b)).unwrap();
        unsafe {self.add_kern.enq().unwrap();}
        self.pq.finish().unwrap();
    }


    pub fn product_buf(&mut self, rows:usize, cols:usize,
                   a:&ocl::Buffer<f32>, b:&ocl::Buffer<f32>, c:&ocl::Buffer<f32>) {
        self.pq.set_dims([rows, cols]);
        self.product_kern.set_default_global_work_size(*self.pq.dims());
        self.product_kern.set_arg("rows", rows as u32).unwrap();
        self.product_kern.set_arg("cols", cols as u32).unwrap();
        self.product_kern.set_arg("A", Some::<&ocl::Buffer<f32>>(&a)).unwrap();
        self.product_kern.set_arg("B", Some::<&ocl::Buffer<f32>>(&b)).unwrap();
        self.product_kern.set_arg("C", Some::<&ocl::Buffer<f32>>(&c)).unwrap();
        unsafe {self.product_kern.enq().unwrap();}
        self.pq.finish().unwrap();
    }


    pub fn update_buf(&mut self, rows:usize, cols:usize, rate:f32,
                   delta:&ocl::Buffer<f32>, input:&ocl::Buffer<f32>) {
        self.pq.set_dims([rows, cols]);
        self.update_kern.set_default_global_work_size(*self.pq.dims());
        self.update_kern.set_arg("rows", rows as u32).unwrap();
        self.update_kern.set_arg("cols", cols as u32).unwrap();
        self.update_kern.set_arg("rate", rate as f32).unwrap();
        self.update_kern.set_arg("delta", Some::<&ocl::Buffer<f32>>(&delta)).unwrap();
        self.update_kern.set_arg("input", Some::<&ocl::Buffer<f32>>(&input)).unwrap();
        unsafe {self.update_kern.enq().unwrap();}
        self.pq.finish().unwrap();
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut gpu = Oclot::new();
        let a = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        let b = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        let expected = Mtx::new((2,2), vec![22., 28., 49., 64.]);
        assert_eq!(expected, gpu.dot(&a, &b));

        let c = Mtx::new((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let d = Mtx::new((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let expected = Mtx::new((3,3), vec![30., 36., 42., 66., 81., 96., 102., 126., 150.]);
        assert_eq!(expected, gpu.dot(&c, &d));

        let e = Mtx::new((1, 1), vec![3.]);
        let f = Mtx::new((1, 1), vec![5.]);
        let expected = Mtx::new((1,1), vec![15.]);
        assert_eq!(expected, gpu.dot(&e, &f));

        let g = Mtx::new((2,4), vec![1.,2.,3.,4., 1.,2.,3.,4.]);
        let h = Mtx::new((4,1), vec![1.,2.,3.,4.]);
        let expected = Mtx::new((2,1), vec![30., 30.]);
        assert_eq!(expected, gpu.dot(&g, &h));
    }

    #[test]
    fn test_sum() {
        let a = mtx!{
            (3, 4);
            [0, 3, 6, 9,
             1, 4, 7, 10,
             2, 5, 8, 11]
        };

        let mut gpu = Oclot::new();

        assert_eq!(gpu.sum_cols(&a), a.sum_cols());
        assert_eq!(gpu.sum_rows(&a), a.sum_rows());
    }

    #[test]
    fn test_transpose() {
        let a = mtx!{
            (3, 4);
            [0, 3, 6, 9,
             1, 4, 7, 10,
             2, 5, 8, 11]
        };

        let mut gpu = Oclot::new();

        assert_eq!(a.trans(), gpu.trans(&a));
    }

    #[test]
    fn test_read_write() {

        let mut gpu = Oclot::new();

        let buf_a = vec![0.0, 0.1, 0.2, 0.3, 0.4];
        let mut buf_b = vec![0.0; 5];
        let gpu_buf = gpu.create_buffer(buf_a.len());

        gpu.write_buffer(&buf_a, &gpu_buf);
        gpu.read_buffer(&gpu_buf, &mut buf_b);

        assert_eq!(buf_a, buf_b);
    }


    #[test]
    fn test_dot_buffer() {
        let mut gpu = Oclot::new();
        let a = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        let b = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        let mut result = vec![0.; a.shape().0*b.shape().1];

        let a_buf = gpu.create_buffer(a.shape().0*a.shape().1);
        let b_buf = gpu.create_buffer(b.shape().0*b.shape().1);
        let mut res_buf = gpu.create_buffer(a.shape().0*b.shape().1);

        gpu.write_buffer(&a.get_raw(), &a_buf);
        gpu.write_buffer(&b.get_raw(), &b_buf);
        gpu.dot_buf(a.shape().0, b.shape().1, a.shape().1, &a_buf, &b_buf, &mut res_buf);
        gpu.read_buffer(&res_buf, &mut result);

        assert_eq!(a.dot(&b).get_raw(), result);
    }


    #[test]
    fn test_sum_buffer() {
        let mut gpu = Oclot::new();
        let a = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        let mut result = vec![0., 0.];

        let a_buf = gpu.create_buffer(6);
        let mut res_buf = gpu.create_buffer(2);

        gpu.write_buffer(&a.get_raw(), &a_buf);
        gpu.sum_buf(0, 2, 3, &a_buf, &mut res_buf);
        gpu.read_buffer(&res_buf, &mut result);

        assert_eq!(a.sum_cols().get_raw(), result);
    }


    #[test]
    fn test_trans_buffer() {
        let mut gpu = Oclot::new();
        let a = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        let mut result = vec![0.; 6];

        let a_buf = gpu.create_buffer(6);
        let mut res_buf = gpu.create_buffer(6);

        gpu.write_buffer(&a.get_raw(), &a_buf);
        gpu.transpose_buf(2, 3, &a_buf, &mut res_buf);
        gpu.read_buffer(&res_buf, &mut result);

        assert_eq!(a.trans().get_raw(), result);
    }


    #[test]
    fn test_add_buffer() {
        let mut gpu = Oclot::new();
        let (rows, cols) = (3, 2);
        let a = Mtx::new((rows, cols), vec![1., 2., 3., 4., 5., 6.]);
        let b = Mtx::new((rows, cols), vec![1., 2., 3., 4., 5., 6.]);
        let mut result = vec![0.; rows*cols];

        let a_buf = gpu.create_buffer(a.size());
        let b_buf = gpu.create_buffer(b.size());

        gpu.write_buffer(&a.get_raw(), &a_buf);
        gpu.write_buffer(&b.get_raw(), &b_buf);
        gpu.add_buf(rows, cols, &a_buf, &b_buf);
        gpu.read_buffer(&b_buf, &mut result);

        assert_eq!(a.add(&b).get_raw(), result);
    }


    #[test]
    fn test_product_buffer() {
        let mut gpu = Oclot::new();
        let (rows, cols) = (3, 2);
        let a = Mtx::new((rows, cols), vec![1., 2., 3., 4., 5., 6.]);
        let b = Mtx::new((rows, cols), vec![1., 2., 3., 4., 5., 6.]);
        let mut result = vec![0.; rows*cols];

        let a_buf = gpu.create_buffer(a.size());
        let b_buf = gpu.create_buffer(b.size());
        let result_buf = gpu.create_buffer(b.size());

        gpu.write_buffer(&a.get_raw(), &a_buf);
        gpu.write_buffer(&b.get_raw(), &b_buf);
        gpu.product_buf(rows, cols, &a_buf, &b_buf, &result_buf);
        gpu.read_buffer(&result_buf, &mut result);

        assert_eq!(a.prod(&b).get_raw(), result);
    }


    #[test]
    fn test_update_buffer() {
        let mut gpu = Oclot::new();
        let (rows, cols) = (3, 2);
        let a = Mtx::new((rows, cols), vec![1., 2., 3., 4., 5., 6.]);
        let b = Mtx::new((rows, cols), vec![1., 2., 3., 4., 5., 6.]);
        let mut result = vec![0.; rows*cols];

        let a_buf = gpu.create_buffer(a.size());
        let b_buf = gpu.create_buffer(b.size());

        gpu.write_buffer(&a.get_raw(), &a_buf);
        gpu.write_buffer(&b.get_raw(), &b_buf);
        gpu.update_buf(rows, cols, 0.5, &a_buf, &b_buf);
        gpu.read_buffer(&b_buf, &mut result);

        assert_eq!(a.add(&b.func(|x|x*0.5)).get_raw(), result);
    }
}
