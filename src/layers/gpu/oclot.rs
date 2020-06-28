extern crate ocl;

use super::Mtx;

#[derive(Debug)]
pub struct Oclot {
    pq: ocl::ProQue,
    trans_kern: ocl::Kernel,
    sum_kern: ocl::Kernel,
    dot_kern: ocl::Kernel,
    forward_kern: ocl::Kernel,
}


impl Oclot {
    pub fn new() -> Oclot {
        let (pq, trans_kern, sum_kern, dot_kern, forward_kern) = Oclot::init_kernels();
        Oclot {
            pq,
            trans_kern,
            sum_kern,
            dot_kern,
            forward_kern,
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


    fn _dot(&mut self, m:usize, n:usize, k:usize, a:&Vec<f32>, b:&Vec<f32>) -> Vec<f32> {
        self.pq.set_dims([m, n]);
        let mut response: Vec<f32> = vec![0.0f32; m*n];

        let device_a = ocl::Buffer::builder()
            .queue(self.pq.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(m*k)
            .copy_host_slice(&a)
            .build().unwrap();

        let device_b = ocl::Buffer::builder()
            .queue(self.pq.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(n*k)
            .copy_host_slice(&b)
            .build().unwrap();

        let device_c: ocl::Buffer<f32> = self.pq.create_buffer().unwrap();

        self.dot_kern.set_default_global_work_size(*self.pq.dims());
        self.dot_kern.set_arg("M", m as u32).unwrap();
        self.dot_kern.set_arg("N", n as u32).unwrap();
        self.dot_kern.set_arg("K", k as u32).unwrap();
        self.dot_kern.set_arg("A", Some::<&ocl::Buffer<f32>>(&device_a)).unwrap();
        self.dot_kern.set_arg("B", Some::<&ocl::Buffer<f32>>(&device_b)).unwrap();
        self.dot_kern.set_arg("C", Some::<&ocl::Buffer<f32>>(&device_c)).unwrap();

        unsafe {self.dot_kern.enq().unwrap();}

        device_c.read(&mut response).enq().unwrap();
        return response;
    }


    fn init_kernels() -> (ocl::ProQue, ocl::Kernel, ocl::Kernel, ocl::Kernel, ocl::Kernel) {
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

        (pq, trans_kern, sum_kern, dot_kern, fw_kern)
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

        let device_c: ocl::Buffer<f32> = self.pq.create_buffer().unwrap();

        self.trans_kern.set_default_global_work_size(*self.pq.dims());
        self.trans_kern.set_arg("rows", rows as u32).unwrap();
        self.trans_kern.set_arg("cols", cols as u32).unwrap();
        self.trans_kern.set_arg("X", Some::<&ocl::Buffer<f32>>(&device_x)).unwrap();
        self.trans_kern.set_arg("C", Some::<&ocl::Buffer<f32>>(&device_c)).unwrap();

        unsafe {self.trans_kern.enq().unwrap();}

        device_c.read(&mut response).enq().unwrap();
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

        let device_c: ocl::Buffer<f32> = self.pq.create_buffer().unwrap();

        self.sum_kern.set_default_global_work_size(*self.pq.dims());
        self.sum_kern.set_arg("dim", dim as u32).unwrap();
        self.sum_kern.set_arg("rows", rows as u32).unwrap();
        self.sum_kern.set_arg("cols", cols as u32).unwrap();
        self.sum_kern.set_arg("X", Some::<&ocl::Buffer<f32>>(&device_x)).unwrap();
        self.sum_kern.set_arg("C", Some::<&ocl::Buffer<f32>>(&device_c)).unwrap();

        unsafe {self.sum_kern.enq().unwrap();}

        device_c.read(&mut response).enq().unwrap();
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
            .flags(ocl::MemFlags::new().read_only())
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
        return response;
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
}
