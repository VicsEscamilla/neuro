extern crate ocl;

use super::Mtx;

#[derive(Debug)]
pub struct Oclot {
    dot_pq: ocl::ProQue,
    dot_kern: ocl::Kernel,
    forward_pq: ocl::ProQue,
    forward_kern: ocl::Kernel
}


impl Oclot {
    pub fn new() -> Oclot {
        let (dot_pq, dot_kern) = Oclot::init_dot();
        let (forward_pq, forward_kern) = Oclot::init_forward();
        Oclot {
            dot_pq,
            dot_kern,
            forward_pq,
            forward_kern,
        }
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
        self.dot_pq.set_dims([m, n]);
        let mut response: Vec<f32> = vec![0.0f32; m*n];

        let device_a = ocl::Buffer::builder()
            .queue(self.dot_pq.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(m*k)
            .copy_host_slice(&a)
            .build().unwrap();

        let device_b = ocl::Buffer::builder()
            .queue(self.dot_pq.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(n*k)
            .copy_host_slice(&b)
            .build().unwrap();

        let device_c: ocl::Buffer<f32> = self.dot_pq.create_buffer().unwrap();

        self.dot_kern.set_default_global_work_size(*self.dot_pq.dims());
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


    fn init_dot() -> (ocl::ProQue, ocl::Kernel){
        static KERNEL_MTXDOT: &'static str = r#"
        __kernel void dot(const int M, const int N, const int K,
                              const __global float* A,
                              const __global float* B,
                              __global float* C) {

            // Thread identifiers
            const int globalRow = get_global_id(0); // Row ID of C (0..M)
            const int globalCol = get_global_id(1); // Col ID of C (0..N)

            // Compute a single element (loop over K)
            float acc = 0.0f;
            for (int k=0; k<K; k++) {
                acc += A[k + globalRow*K] * B[globalCol + N*k];
            }

            // Store the result
            C[globalCol + globalRow*N] = acc;
        }
        "#;

        let pq = ocl::ProQue::builder()
                    .src(KERNEL_MTXDOT)
                    .build().unwrap();

        let kern = pq.kernel_builder("dot")
            .arg_named("M", 1 as u32)
            .arg_named("N", 1 as u32)
            .arg_named("K", 1 as u32)
            .arg_named("A", None::<&ocl::Buffer<f32>>)
            .arg_named("B", None::<&ocl::Buffer<f32>>)
            .arg_named("C", None::<&ocl::Buffer<f32>>)
            .build().unwrap();

        (pq, kern)
    }

    fn init_forward() -> (ocl::ProQue, ocl::Kernel){
        static KERNEL_FORWARD: &'static str = r#"
        __kernel void forward(const int M, const int N, const int K,
                              const __global float* X,
                              const __global float* W,
                              const __global float* B,
                              __global float* R) {

            // Thread identifiers
            const int globalRow = get_global_id(0); // Row ID of C (0..M)
            const int globalCol = get_global_id(1); // Col ID of C (0..N)

            // Compute a single element (loop over K)
            float acc = 0.0f;
            for (int k=0; k<K; k++) {
                acc += X[k + globalRow*K] * W[globalCol + N*k];
            }

            // Store the result
            R[globalCol + globalRow*N] = acc + B[globalCol];
        }
        "#;

        let pq = ocl::ProQue::builder()
                    .src(KERNEL_FORWARD)
                    .build().unwrap();

        let kern = pq.kernel_builder("forward")
            .arg_named("M", 1 as u32)
            .arg_named("N", 1 as u32)
            .arg_named("K", 1 as u32)
            .arg_named("X", None::<&ocl::Buffer<f32>>)
            .arg_named("W", None::<&ocl::Buffer<f32>>)
            .arg_named("B", None::<&ocl::Buffer<f32>>)
            .arg_named("R", None::<&ocl::Buffer<f32>>)
            .build().unwrap();

        (pq, kern)
    }


    fn _forward(&mut self, m:usize, n:usize, k:usize, x:&Vec<f32>, w:&Vec<f32>, b:&Vec<f32>) -> Vec<f32> {
        self.forward_pq.set_dims([m, n]);
        let mut response: Vec<f32> = vec![0.0f32; m*n];

        let device_x = ocl::Buffer::builder()
            .queue(self.forward_pq.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(m*k)
            .copy_host_slice(&x)
            .build().unwrap();

        let device_w = ocl::Buffer::builder()
            .queue(self.forward_pq.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(n*k)
            .copy_host_slice(&w)
            .build().unwrap();

        let device_b = ocl::Buffer::builder()
            .queue(self.forward_pq.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(n)
            .copy_host_slice(&b)
            .build().unwrap();

        let device_r = ocl::Buffer::builder()
            .queue(self.forward_pq.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(m*n)
            .build().unwrap();

        self.forward_kern.set_default_global_work_size(*self.forward_pq.dims());
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
}
