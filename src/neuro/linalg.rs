use ndarray::prelude::*;

#[derive(Debug, PartialEq, Clone)]
pub struct Mtx {
    array: Array2<f64>
}

impl Mtx {
    pub fn new(shape: (usize, usize), raw: Vec<f64>) -> Self {
        if shape.0 * shape.1 != raw.len() {
            panic!("invalid shape");
        }
        Mtx {array: Array::from_shape_vec(shape, raw).unwrap()}
    }

    pub fn trans(&self) -> Self {
        Mtx {array:self.array.clone().reversed_axes()}
    }

    pub fn show(&self) {
        println!("{:#?}", self.array);
    }

    pub fn shape(&self) -> (usize, usize) {
        let shape = self.array.shape();
        (shape[0], shape[1])
    }

    pub fn add(&self, other: &Self) -> Self {
        if self.array.shape() != other.array.shape() {
            panic!("invalid shape");
        }

        Mtx {
            array: &self.array + &other.array
        }
    }

    pub fn add_vector(&self, vector: &Vec<f64>) -> Self {
        if vector.len() != self.array.shape()[1] {
            panic!("invalid shape");
        }

        Mtx {array: &self.array + &Array::from_shape_vec((1, vector.len()), vector.to_vec()).unwrap()}
    }

    pub fn dot(&self, other: &Self) -> Self {
        if self.array.shape()[1] != other.array.shape()[0] {
            panic!("invalid shape");
        }
        Mtx {array:self.array.dot(&other.array)}
    }

    pub fn func<F: Fn(&f64)->f64>(&self, f: F) -> Self {
        Mtx {array: self.array.map(|x| f(x))}
    }

    pub fn prod(&self, other: &Self) -> Self {
        if self.array.shape() != other.array.shape() {
            panic!("invalid shape");
        }

        Mtx {array: &self.array * &other.array}
    }

    pub fn sum(&self, dim: usize) -> Self {
        if dim >= 2 {
            panic!("invalid dimension");
        }

        let (rows, cols) = self.shape();

        if dim == 0 {
            Mtx{array:Array::from_shape_vec((1, cols), self.array.sum_axis(Axis(dim)).as_slice().unwrap().to_vec()).unwrap()}
        } else {
            Mtx{array:Array::from_shape_vec((rows, 1), self.array.sum_axis(Axis(dim)).as_slice().unwrap().to_vec()).unwrap()}
        }
    }

    pub fn get_raw(&self) -> Vec<f64> {
        return self.array.as_slice().unwrap().to_vec();
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
        assert_eq!(a.sum(1), expected);

        let a = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        let expected = Mtx::new((1, 2), vec![9., 12.]);
        assert_eq!(a.sum(0), expected);
    }

    #[test]
    fn test_get_raw() {
        let a = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        let expected = vec![1., 2., 3., 4., 5., 6.];
        assert_eq!(a.get_raw(), expected);

        let a = Mtx::new((3, 2), vec![1., 2., 3., 4., 5., 6.]);
        let expected = Mtx::new((1, 2), vec![9., 12.]);
        assert_eq!(a.sum(0), expected);
    }
}
