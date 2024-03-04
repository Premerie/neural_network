use rand::prelude::*;
use crate::matrix::*;

#[derive(Debug)]
pub enum Activation {
    Tanh,
    Sigmoid,
    ReLu,
}

#[derive(Debug)]
pub struct NN {
    learning_rate: f64,
    max_epochs: usize,
    max_errors: f64,
    activation_func: Activation,
    arch: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    activations: Vec<Matrix>,
}

pub struct NN_Config {
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub max_errors: f64,
}

impl NN_Config {
    pub fn new(learning_rate: f64, max_epochs: usize, max_errors: f64) -> Self {
        NN_Config {
            learning_rate,
            max_epochs,
            max_errors,
        }
    }
    
    pub fn init() -> Self {
        NN_Config {
            learning_rate: 0.05,
            max_epochs: 100000,//usize::MAX,
            max_errors: 0.0001,
        }
    }
}

impl NN {
    pub fn new(activation_func: Activation, arch: Vec<usize>, options: NN_Config) -> Self {
        let size: usize = arch.len() - 1;
        NN {
            learning_rate: options.learning_rate,
            max_epochs: options.max_epochs,
            max_errors: options.max_errors,
            activation_func,
            arch,
            weights: vec![Matrix::init(1, 1); size],
            biases: vec![Matrix::init(1, 1); size],
            activations: vec![Matrix::init(1, 1); size + 1],
        }
    }
    
    pub fn init(&mut self, low: f64, high: f64) {
        for i in 1..self.arch.len() {
            self.weights[i - 1] = Matrix::init(self.arch[i], self.arch[i - 1])
                .random(low, high);
            self.biases[i - 1] = Matrix::init(self.arch[i], 1)
                .random(low + 0.3, high - 0.3);
        }
    }
    
    pub fn forward_tanh(&mut self, input: Matrix) -> &Matrix {
        self.activations[0] = input;
        for i in 0..self.weights.len() {
            self.activations[i + 1] = self.weights[i]
                .dot(&self.activations[i])
                .add(&self.biases[i])
                .tanh();
        }
        &self.activations[self.weights.len()]
    }
    
    pub fn forward_sigmoid(&mut self, input: Matrix) -> &Matrix {
        self.activations[0] = input;
        for i in 0..self.weights.len() {
            self.activations[i + 1] = self.weights[i]
                .dot(&self.activations[i])
                .add(&self.biases[i])
                .sigmoid();
        }
        &self.activations[self.weights.len()]
    }
    
    pub fn forward_relu(&mut self, input: Matrix) -> &Matrix {
        self.activations[0] = input;
        for i in 0..self.weights.len() {
            self.activations[i + 1] = self.weights[i]
                .dot(&self.activations[i])
                .add(&self.biases[i])
                .relu();
        }
        &self.activations[self.weights.len()]
    }
    
    pub fn forward(&mut self, input: Matrix) -> &Matrix {
        
        let out: &Matrix;
        
        match self.activation_func {
            Activation::Tanh => out = self.forward_tanh(input),
            Activation::Sigmoid => out = self.forward_sigmoid(input),
            Activation::ReLu => out = self.forward_relu(input),
        }
        out
    }
    
    pub fn predict(&mut self, input: Matrix) -> Matrix {
        self.forward(input.transpose()).transpose()
    }
    
    pub fn fit(&mut self, input: Matrix, output: Matrix) {
        let x = input.transpose();
        let y = output.transpose();
        let mut delta: Matrix;
        let mut error: f64 = 0.;
        let mut count = 0;
        for _ in 0..self.max_epochs {
            let mut vector: Vec<f64> = Vec::new();
            for j in 0..x.cols {
                //forward propagation
                delta = self.forward(x.get_col(j)).minus(&y.get_col(j));
                
                vector.push(delta.get(0, 0));
                
                //backward propagation
                self.gradient_descent(&mut delta.mul_scalar(2.0));
            }
            error = Matrix::new(y.rows, y.cols, vector).cost();
            if error < self.max_errors {break;}
            count += 1;
        }
        println!("error: {}", error);
        println!("count: {}", count);
    }
    
    pub fn grad_sig(&mut self, delta: &mut Matrix) {
        let idx: usize = self.weights.len();
            
        for i in (0..idx).rev() {
            
            if i != (idx - 1) {
                *delta = self.weights[i + 1]
                .transpose()
                .dot(delta)
                .mul(&self.activations[i + 1]
                    .sigmoid_derivative()
                );
            }
              
            self.weights[i] = self.weights[i]
                .add(&delta
                    .dot(&self.activations[i]
                        .transpose()
                    ).mul_scalar(-self.learning_rate)
                );
                
            self.biases[i] = self.biases[i]
                .add(&delta
                    .mul_scalar(-self.learning_rate)
                );
        }
    }
    
    pub fn grad_tanh(&mut self, delta: &mut Matrix) {
        let idx: usize = self.weights.len();
            
        for i in (0..idx).rev() {
            
            if i != (idx - 1) {
                *delta = self.weights[i + 1]
                .transpose()
                .dot(delta)
                .mul(&self.activations[i + 1]
                    .tanh_derivative()
                );
            }
              
            self.weights[i] = self.weights[i]
                .add(&delta
                    .dot(&self.activations[i]
                        .transpose()
                    ).mul_scalar(-self.learning_rate)
                );
                
            self.biases[i] = self.biases[i]
                .add(&delta
                    .mul_scalar(-self.learning_rate)
                );
        }
    }
    
    pub fn grad_relu(&mut self, delta: &mut Matrix) {
        let idx: usize = self.weights.len();
            
        for i in (0..idx).rev() {
            
            if i != (idx - 1) {
                *delta = self.weights[i + 1]
                .transpose()
                .dot(delta)
                .mul(&self.activations[i + 1]
                    .relu_derivative()
                );
            }
              
            self.weights[i] = self.weights[i]
                .add(&delta
                    .dot(&self.activations[i]
                        .transpose()
                    ).mul_scalar(-self.learning_rate)
                );
                
            self.biases[i] = self.biases[i]
                .add(&delta
                    .mul_scalar(-self.learning_rate)
                );
        }
    }
    
    pub fn gradient_descent(&mut self, delta: &mut Matrix) {
        match self.activation_func {
            Activation::Tanh => self.grad_tanh(delta),
            Activation::Sigmoid => self.grad_sig(delta),
            Activation::ReLu => self.grad_relu(delta),
        }
    }
    
    pub fn show(&self) {
        println!("{:#?}", self);
    }
}