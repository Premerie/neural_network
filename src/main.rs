mod nn;
mod matrix;

use nn::*;
use matrix::*;

fn main() {
    let input: Matrix = Matrix::new( 4, 2, vec![1., 0., 0., 1., 1., 1., 0., 0.]);
    
    let output: Matrix = Matrix::new( 4, 1, vec![1., 1., 1., 0.]);
    
    let options: NN_Config = NN_Config::init();
    
    let mut nn: NN = NN::new(
        Activation::Tanh, 
        vec![2, 4, 2, 1],
        options
    );
    
    nn.init(-0.5, 0.5);
    //nn.show();
    //let train_input = input.normalize_cols();
    
    nn.fit(input.clone(), output.clone());
    nn.show();
    
    
    for i in 0..input.rows {
        input.get_row(i).print();
        nn.predict(input.get_row(i)).print();
    }
}