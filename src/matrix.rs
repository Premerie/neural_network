use rand::prelude::*;
use std::f64::consts::E;

#[derive(Clone)]
#[derive(Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub matrix: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, matrix: Vec<f64>) -> Self {
        Matrix { rows, cols, matrix }
    }

    pub fn init(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            matrix: vec![0.; rows * cols],
        }
    }
    
    pub fn max(left: f64, right: f64) -> f64 {
        if left < right {
            return right;
        }
        left
    }
    
    pub fn rel_deriv(left: f64, right: f64) -> f64 {
        if left < right {
            return 1.;
        }
        0.
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.matrix[row * self.cols + col]
    }
    
    pub fn get_col(&self, col: usize) -> Matrix {
        let mut out: Matrix = Matrix::init(self.rows, 1);
        for row in 0..self.rows {
            out.matrix[row * out.cols] = self.get(row, col);
        }
        out
    }
    
    pub fn get_row(&self, row: usize) -> Matrix {
        let mut out: Matrix = Matrix::init(1, self.cols);
        for col in 0..self.cols {
            out.matrix[col] = self.get(row, col);
        }
        out
    }

    pub fn identity(&self) -> Matrix {
        let mut out: Matrix = Matrix::init(self.rows, self.cols);  

        for i in 0..self.rows {
            for j in 0..self.cols {
                if i == j {
                    out.matrix[i * out.cols + j] = 1.;
                }
            }
        }

        out
    }

    pub fn random(&self, low: f64, high: f64) -> Matrix {
        let mut matrix: Vec<f64> = vec![0.; self.matrix.len()];

        let mut rng = rand::thread_rng();

        for i in 0..self.rows {
            for j in 0..self.cols {
                matrix[i * self.cols + j] = rng.gen::<f64>() * (high - low) + low;
            }
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            matrix,
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut matrix: Vec<f64> = vec![0.; self.matrix.len()];
        for i in 0..self.rows {
            for j in 0..self.cols {
                matrix[j * self.rows + i] = self.get(i, j);
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            matrix,
        }
    }
    
    pub fn normalize_cols(&self) -> Self {
        
        let mut v: Vec<f64>;
        
        let mut magnitudes: Vec<f64> = Vec::new();
        
        let mut matrix: Vec<f64> = vec![0.; self.matrix.len()];
        
        //magnitude
        for i in 0..self.cols {
            let mut sum: f64 = 0.0;
            v = self.get_col(i).matrix;
            for element in v.iter() {
                sum += element.powf(2.0);
            } 
            magnitudes.push(sum.sqrt());
        }
        
        //normalize
        for i in 0..self.cols {
            v = self.get_col(i).matrix;
            let mut row: usize = 0;
            for element in v.iter() {
                matrix[row * self.cols + i] = element / magnitudes[i];
                row += 1;
            } 
        }
        
        Matrix {
            rows: self.rows,
            cols: self.cols,
            matrix,
        }
    }
    
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.cols);
        assert_eq!(self.rows, other.rows);
        
        let mut out: Matrix = Matrix::init(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                out.matrix[i * out.cols + j] = self.get(i, j) + other.get(i, j);
            }
        }
        out
    }
    
    pub fn minus(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.cols);
        assert_eq!(self.rows, other.rows);
        
        let mut out: Matrix = Matrix::init(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                out.matrix[i * out.cols + j] = self.get(i, j) - other.get(i, j);
            }
        }
        out
    }
    
    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.cols);
        assert_eq!(self.rows, other.rows);
        
        let mut out: Matrix = Matrix::init(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                out.matrix[i * out.cols + j] = self.get(i, j) * other.get(i, j);
            }
        }
        out
    }
    
    pub fn mul_scalar(&self, other: f64) -> Matrix {
        
        let mut out: Matrix = Matrix::init(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                out.matrix[i * out.cols + j] = self.get(i, j) * other;
            }
        }
        out
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        
        assert_eq!(self.cols, other.rows);
        
        let mut out: Matrix = Matrix::init(self.rows, other.cols);

        let mut sum: f64 = 0.;

        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                out.matrix[i * out.cols + j] = sum;
                sum = 0.;
            }
        }
        out
    }
    
    pub fn sigmoid(&self) -> Matrix {
        let mut out: Matrix = Matrix::init(self.rows, self.cols);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.matrix[i * out.cols + j] = 1.0 / (1.0 + E.powf(-self.get(i, j)));
            }
        }
        out
    }
    
    pub fn sigmoid_derivative(&self) -> Matrix {
        let mut out: Matrix = Matrix::init(self.rows, self.cols);
        let mut sig: f64;
        for i in 0..self.rows {
            for j in 0..self.cols {
                sig = self.get(i, j);
                out.matrix[i * out.cols + j] = sig * (1.0 - sig);
            }
        }
        out
    }
    
    pub fn tanh(&self) -> Matrix {
        let mut out: Matrix = Matrix::init(self.rows, self.cols);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.matrix[i * out.cols + j] = self.get(i, j).cos();//.tanh();
            }
        }
        out
    }
    
    pub fn tanh_derivative(&self) -> Matrix {
        let mut out: Matrix = Matrix::init(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.matrix[i * out.cols + j] = -self.get(i, j).sin();//1.0 - self.get(i, j).powf(2.0);
            }
        }
        out
    }
    
    pub fn relu(&self) -> Matrix {
        let mut out: Matrix = Matrix::init(self.rows, self.cols);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.matrix[i * out.cols + j] = Matrix::max(0., self.get(i, j));
            }
        }
        out
    }
    
    pub fn relu_derivative(&self) -> Matrix {
        let mut out: Matrix = Matrix::init(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.matrix[i * out.cols + j] = Matrix::rel_deriv(0., self.get(i, j));
            }
        }
        out
    }
    
    pub fn cost(&self) -> f64 {
        let mut sum: f64 = 0.;
        let size: f64 = self.matrix.len() as f64;
        for i in 0..self.rows {
            for j in 0..self.cols {
                sum += self.get(i, j).powf(2.) / size;
            }
        }
        sum.sqrt()
    }

    pub fn print(&self) {
        println!("[");
        for i in 0..self.rows {
            print!("    [\n        ");
            for j in 0..self.cols {
                print!("{}    ", self.get(i, j));
            }
            print!("\n    ]\n");
        }
        println!("\n]");
    }

    pub fn show(&self) {
        println!("{:#?}", &self);
    }
}
