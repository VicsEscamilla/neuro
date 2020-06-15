use super::{Mtx, Layer};

#[derive(Clone)]
pub struct Softmax {
}

impl Softmax {
    pub fn new() -> Box<Self> {
        Box::new(Softmax {})
    }
}

impl Layer for Softmax {
    fn forward(&mut self, x: &Mtx) -> Mtx {
        x.softmax()
    }


    fn backward(&mut self, x: &Mtx, delta:&Mtx) -> Mtx {
        if x.shape() != delta.shape() {
            panic!("Softmax can only be used in the last layer");
        }

        let (rows, _) = x.shape();
        let x_raw = &x.get_raw();
        let delta_raw = &delta.get_raw();

        Mtx::new(x.shape(),
            (0..x_raw.len())
                .map(|i| {
                    if i==0 || rows % i == 0 {
                        x_raw[i] * (1.0-delta_raw[i])
                    }
                    else {
                        -x_raw[i] * delta_raw[i]
                    }
                }).collect())
        /*
    for i in range(len(self.value)):
        for j in range(len(self.input)):
            if i == j:
                self.gradient[i] = self.value[i] * (1-self.input[i))
            else:
                 self.gradient[i] = -self.value[i]*self.input[j]
         * */
    }


    fn update(&mut self, _rate: f32) {
    }


    fn initialize(&mut self, _input_size: usize) {
    }

    fn input_size(&self) -> usize {
        0
    }

    fn error(&self, result: &Mtx, y: &Mtx) -> Mtx {
        result.func(|&x|-x*x).add(&y.func(|&x| x*x))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_softmax() {
        let x = Softmax::new();
        //assert_eq!(a.add(&b), expected);
    }

}
