use std::f32::consts::E;
use serde::{Serialize, Deserialize};

#[macro_export]
macro_rules! mtx {
    (($rows:expr , $cols:expr); &$x:expr) => {
        Mtx::new(($rows, $cols), $x.to_vec())
    };
    (($rows:expr , $cols:expr); [$( $x:expr ),*]) => {
        Mtx::new(($rows, $cols), vec![$( $x as f32 ),*])
    };
    (($rows:expr , $cols:expr); [$( $x:expr ),+,]) => {
        Mtx::new(($rows, $cols), vec![$( $x as f32 ),*])
    };
    (($rows:expr , $cols:expr); $elem:expr) => {
        Mtx::new(($rows, $cols), vec![$elem as f32; $rows*$cols])
    }
}

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
        Mtx {
            shape: (self.shape.1, self.shape.0),
            raw: (0..self.shape.1)
                    .flat_map(|i| {
                        (0..self.shape.0)
                            .map(move |j| self.raw[j*self.shape.1 + i])
                    }).collect()
        }
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

    pub fn add_vector(&self, vec: &Vec<f32>) -> Self {
        if vec.len() != self.shape.1 {
            panic!("invalid shape");
        }

        Mtx {
            shape: self.shape,
            raw: (0..self.raw.len())
                    .map(|i| self.raw[i] + vec[i%vec.len()])
                    .collect()
        }
    }

    pub fn dot(&self, other: &Self) -> Self {
        if self.shape.1 != other.shape.0 {
            panic!("invalid shape");
        }

        Mtx {
            shape: (self.shape.0, other.shape.1),
            raw: (0..self.shape.0).flat_map(|i| {
                        (0..other.shape.1).map(move |j| {
                            (0..self.shape.1).map(|k| {
                                let a = i*self.shape.1+k;
                                let b = k*other.shape.1+j;
                                self.raw[a] * other.raw[b]
                            }).sum()
                        })
                    }).collect()
        }
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
            Mtx {
                shape:(rows, 1),
                raw: (0..rows).map(|i| {
                        (0..cols).map(move |j| self.raw[i*cols + j]).sum()
                    }).collect()
            }
        } else {
            Mtx {
                shape:(1, cols),
                raw: (0..cols).map(|j| {
                        (0..rows).map(move |i| self.raw[i*cols + j]).sum()
                    }).collect()
            }
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
        let cols = self.shape().1;
        Mtx {
            shape:self.shape,
            raw: index.iter().flat_map(|i| {
                        (0..cols).map(move |j| self.raw[i*cols + j])
                    }).collect()
        }
    }

    pub fn softmax(&self) -> Self {
        let (rows, cols) = self.shape();

        Mtx {
            shape: self.shape,
            raw: (0..rows).flat_map(|i| {
                    let sum: f32 = (&self.raw[i*cols..(i+1)*cols])
                                    .iter()
                                    .map(|x: &f32| E.powf(*x))
                                    .sum();
                    (0..cols).map(move |j| E.powf(self.raw[i*cols + j])/sum)
                }).collect()
        }
    }
}


#[cfg(test)]
mod tests {
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
        let index = vec![1, 3, 2, 0];

        assert_eq!(a.reorder_rows(&index).get_raw(),
            vec![3., 4., 7., 8., 5., 6., 1., 2.]);
        assert_eq!(b.reorder_rows(&index).get_raw(), vec![3., 7., 5., 1.]);
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

    #[test]
    fn test_mtx() {
        let expected = Mtx::new((2,3), vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(expected, mtx![(2,3); [1,2,3,4,5,6]]);
        assert_eq!(expected, mtx![(2,3); [1,2,3,4,5,6,]]);

        let expected = Mtx::new((20,5), vec![3.14; 100]);
        assert_eq!(expected, mtx![(20,5); 3.14]);

        let expected = Mtx::new((2,2), vec![1., 2., 3., 4.]);
        let vec = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        assert_eq!(expected, mtx![(2, 2); &vec[0..4]]);
    }
}
