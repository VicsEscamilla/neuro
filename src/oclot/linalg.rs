use std::f32::consts::E;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, PartialEq, PartialOrd, Clone)]
pub struct Mtx {
    shape: (usize, usize),
    raw: Vec<f32>
}

impl Mtx {
    pub fn new(shape: (usize, usize), raw: Vec<f32>) -> Self {
        if shape.0 * shape.1 != raw.len() {
            panic!("invalid shape");
        }
        Mtx{shape, raw}
    }

    pub fn trans(&self) -> Self {
        let shape = (self.shape.1, self.shape.0);
        let mut raw: Vec<f32> = Vec::with_capacity(self.raw.len());
        for i in 0..self.shape.1 {
            for j in 0..self.shape.0 {
                raw.push(self.raw[j*self.shape.1 + i]);
            }
        }

        Mtx {shape, raw}
    }

    pub fn show(&self) {
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                print!("{:?} ", self.raw[i*self.shape.1 + j]);
            }
            println!();
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        return self.shape;
    }

    pub fn add(&self, other: &Self) -> Self {
        if self.shape != other.shape {
            panic!("invalid shape");
        }

        Mtx {
            shape: (self.shape.0, self.shape.1),
            raw: self.raw.iter()
                .zip(&other.raw)
                .map(|(&a, &b)| a + b)
                .collect()
        }
    }

    pub fn add_vector(&self, vector: &Vec<f32>) -> Self {
        if vector.len() != self.shape.1 {
            panic!("invalid shape");
        }

        let mut raw: Vec<f32> = Vec::with_capacity(self.raw.len());
        for i in 0..self.raw.len() {
            raw.push(self.raw[i] + vector[i%vector.len()]);
        }

        Mtx {shape: self.shape, raw}
    }

    pub fn dot(&self, other: &Self) -> Self {
        if self.shape.1 != other.shape.0 {
            panic!("invalid shape");
        }

        let shape = (self.shape.0, other.shape.1);
        let mut raw: Vec<f32> = Vec::with_capacity(shape.0 * shape.1);
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                let mut sum = 0.;
                for k in 0..self.shape.1 {
                    let indexa = i*self.shape.1+k;
                    let indexb = k*other.shape.1+j;
                    sum += self.raw[indexa] * other.raw[indexb];
                }
                raw.push(sum);
            }
        }
        // let raw = ocl_dot();
        Mtx {shape, raw}
    }

    pub fn func<F: Fn(&f32)->f32>(&self, f: F) -> Self {
        Mtx {
            shape: self.shape,
            raw: self.raw.iter().map(|x| f(x)).collect()
        }
    }

    pub fn prod(&self, other: &Self) -> Self {
        if self.shape != other.shape {
            panic!("invalid shape");
        }

        Mtx {
            shape: (self.shape.0, self.shape.1),
            raw: self.raw.iter()
                .zip(&other.raw)
                .map(|(&a, &b)| a * b)
                .collect()
        }
    }

    pub fn sum(&self, dim: usize) -> Self {
        if dim >= 2 {
            panic!("invalid dimension");
        }

        let (rows, cols) = self.shape();
        if dim == 0 {
            let mut raw: Vec<f32> = Vec::with_capacity(rows);
            for i in 0..rows {
                let mut sum = 0.;
                for j in 0..cols {
                    sum  += self.raw[i*cols + j];
                }
                raw.push(sum);
            }
            Mtx{shape:(rows, 1), raw}
        } else {
            let mut raw: Vec<f32> = Vec::with_capacity(cols);
            for j in 0..cols {
                let mut sum = 0.;
                for i in 0..rows {
                    sum  += self.raw[i*cols + j];
                }
                raw.push(sum);
            }
            Mtx{shape:(1, cols), raw}
        }
    }

    pub fn get_raw(&self) -> Vec<f32> {
        return self.raw.clone();
    }

    pub fn get_row(&self, index: usize) -> Self {
        let (rows, cols) = self.shape();
        if index >= rows {
            panic!("invalid row");
        }
        let i = index*cols;
        return Mtx::new((1, cols), self.raw[i..i+cols].to_vec());
    }

    pub fn reorder_rows(&self, index: &Vec<usize>) -> Self {
        let (rows, cols) = self.shape();
        let mut raw: Vec<f32> = Vec::with_capacity(rows * cols);
        for i in index {
            for j in 0..cols {
                raw.push(self.raw[i*cols + j]);
            }
        }

        Mtx {shape:self.shape, raw}
    }

    pub fn softmax(&self) -> Self {
        let (rows, cols) = self.shape();
        let mut raw: Vec<f32> = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            let sum: f32 = (&self.raw[i*cols..(i+1)*cols]).iter()
                        .map(|x: &f32| E.powf(*x)).sum();
            for j in 0..cols {
                raw.push(E.powf(self.raw[i*cols + j])/sum);
            }
        }
        Mtx {shape:self.shape, raw}
    }
}


#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use rand::seq::SliceRandom;
    use super::*;

    #[test]
    #[should_panic]
    fn test_wrong_shape_01() {
        Mtx::new((2, 1), vec![1., 2., 3., 4.]);
    }

    #[test]
    #[should_panic]
    fn test_wrong_shape_02() {
        Mtx::new((2, 1), vec![1.]);
    }

    #[test]
    #[should_panic]
    fn test_addition_wrong_shape() {
        let a = Mtx::new((2, 1), vec![1., 2.]);
        let b = Mtx::new((2, 2), vec![1., 2., 3., 4.]);
        a.add(&b);
    }

    #[test]
    fn test_addition() {
        let a = Mtx::new((1, 3), vec![1., 2., 3.]);
        let b = Mtx::new((1, 3), vec![2., 4., 6.]);
        let expected = Mtx::new((1, 3), vec![3., 6., 9.]);
        assert_eq!(a.add(&b), expected);
    }

    #[test]
    #[should_panic]
    fn test_addition_vector_wrong_shape() {
        let a = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        a.add_vector(&vec![1., 2., 3., 4.]);
    }

    #[test]
    fn test_addition_vector() {
        let a = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        let expected = Mtx::new((2, 3), vec![2., 4., 6., 5., 7., 9.]);
        assert_eq!(a.add_vector(&vec![1., 2., 3.]), expected);
    }

    #[test]
    fn test_show() {
        Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]).show();
    }

    #[test]
    #[should_panic]
    fn test_product_dot_wrong_shape() {
        let a = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        let b = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        a.dot(&b);
    }

    #[test]
    fn test_product_dot() {
        let a = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        let b = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        let expected = Mtx::new((2, 2), vec![22., 28., 49., 64.]);
        assert_eq!(a.dot(&b), expected);
    }

    #[test]
    fn test_product_scalar() {
        let a = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        let expected = Mtx::new((2, 3), vec![10., 20., 30., 40., 50., 60.]);
        assert_eq!(a.func(|x| x*10.), expected);
    }

    #[test]
    #[should_panic]
    fn test_product_wrong_shape() {
        let a = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        let b = Mtx::new((2, 4), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        a.prod(&b);
    }

    #[test]
    fn test_product() {
        let a = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        let b = Mtx::new((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        let expected = Mtx::new((2, 3), vec![1., 4., 9., 16., 25., 36.]);
        assert_eq!(a.prod(&b), expected);
    }

    #[test]
    fn test_transpose() {
        let a = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        let expected = Mtx::new((2, 3), vec![1., 3., 5., 2., 4., 6.]);
        a.show();
        expected.show();
        assert_eq!(a.trans(), expected);
    }

    #[test]
    fn test_shape() {
        let a = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(a.shape(), (3,2));
    }

    #[test]
    fn test_func() {
        let a = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        let expected = Mtx::new((3, 2), vec![1., 4., 9., 16., 25., 36.]);
        assert_eq!(a.func(|x| x*x), expected);
    }

    #[test]
    #[should_panic]
    fn test_sum_wrong_shape() {
        let a = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        a.sum(2);
    }

    #[test]
    fn test_sum() {
        let a = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        let expected = Mtx::new((3, 1), vec![3., 7., 11.]);
        assert_eq!(a.sum(0), expected);

        let a = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        let expected = Mtx::new((1, 2), vec![9., 12.]);
        assert_eq!(a.sum(1), expected);
    }

    #[test]
    fn test_get_raw() {
        let a = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        let expected = vec![1., 2., 3., 4., 5., 6.];
        assert_eq!(a.get_raw(), expected);

        let a = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        let expected = Mtx::new((1, 2), vec![9., 12.]);
        assert_eq!(a.sum(1), expected);
    }

    #[test]
    fn test_shuffle() {

        let a = Mtx::new((4, 2), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Mtx::new((4, 1), vec![1., 3., 5., 7.]);

        let mut index: Vec<usize> = (0..4).collect();
        index.shuffle(&mut thread_rng());

        // assert this
        a.reorder_rows(&index).show();
        b.reorder_rows(&index).show();
    }

    #[test]
    fn test_softmax() {
        // assert this...
        let a = Mtx::new((1, 8), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        a.softmax().show();
    }

    #[test]
    fn test_get_row() {
        let a = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(a.get_row(0).get_raw(), vec![1., 2.]);
        assert_eq!(a.get_row(1).get_raw(), vec![3., 4.]);
        assert_eq!(a.get_row(2).get_raw(), vec![5., 6.]);
    }
}
