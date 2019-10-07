extern crate ocl;

mod linalg;

pub use linalg::Mtx;

#[derive(Debug)]
pub struct Oclote {
    pq: ocl::ProQue,
    kern: ocl::Kernel
}

impl Oclote {
    pub fn new() -> Oclote {
        static KERNEL_MTXDOT: &'static str = r#"
        __kernel void dot(const int M, const int N, const int K,
                              const __global float* A,
                              const __global float* B,
                              __global float* C) {

            const int row = get_global_id(1);
            const int col = get_global_id(0);

            float acc = 0.0f;
            for (int k=0; k<K; k++) {
                acc += A[k*M + row] * B[col*K + k];
            }
            C[col*M + row] = acc;
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

        Oclote {pq, kern}
    }

    pub fn dot(&mut self, a:&Mtx, b:&Mtx) -> Mtx {
        Mtx::new(
            (a.shape().0, b.shape().1),
            self._dot(a.shape().0, b.shape().1, a.shape().1,
                      &a.get_raw(), &b.get_raw())
        )
    }

    fn _dot(&mut self, m:usize, n:usize, k:usize, a:&Vec<f32>, b:&Vec<f32>) -> Vec<f32> {
        let size: usize = m*n;
        self.pq.set_dims((m, n));
        let mut response: Vec<f32> = vec![0.0f32; size];

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

        self.kern.set_default_global_work_size(*self.pq.dims());
        self.kern.set_arg("M", m as u32).unwrap();
        self.kern.set_arg("N", n as u32).unwrap();
        self.kern.set_arg("K", k as u32).unwrap();
        self.kern.set_arg("A", Some(&device_a)).unwrap();
        self.kern.set_arg("B", Some(&device_b)).unwrap();
        self.kern.set_arg("C", Some(&device_c)).unwrap();

        unsafe {self.kern.enq().unwrap();}

        device_c.read(&mut response).enq().unwrap();
        return response;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut gpu = Oclote::new();
        let a = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        let b = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        let expected = Mtx::new((2,2), vec![22., 28., 49., 64.]);
        assert_eq!(expected, gpu.dot(&a, &b));

        let a = Mtx::new((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let b = Mtx::new((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let expected = Mtx::new((3,3), vec![30., 36., 42., 66., 81., 96., 102., 126., 150.]);
        assert_eq!(expected, gpu.dot(&a, &b));
    }
}
