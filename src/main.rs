use crate::model::train_transformer;
use candle_core::{Device, Tensor};
use smtlib::{backend::z3_binary::Z3Binary, prelude::*, Int, SatResultWithModel, Solver, Storage};

mod model;
mod smt_solver;

// Inspired by 1d cellular automata, here we have a simple rule: x1 XOR x3 XOR x5
// We are interested in veifying, e.g. the correct attention of the trained model
// on the relevant cells (x1, x3, x5) when predicting the next
fn generate_7cell_data() -> (Vec<[f32; 7]>, Vec<f32>) {
    let mut inputs = Vec::new();
    let mut next_states = Vec::new();
    for n in 0..128 {
        let mut neighborhood = [0u8; 7];
        for i in 0..7 {
            neighborhood[i] = ((n >> i) & 1) as u8;
        }
        // Toy rule: next_state = x1 XOR x3 XOR x5
        let next_state = neighborhood[1] ^ neighborhood[3] ^ neighborhood[5];

        let neighborhood = neighborhood.map(|x| x as f32);
        inputs.push(neighborhood);
        next_states.push(next_state as f32);
    }
    (inputs, next_states)
}

fn main() -> anyhow::Result<()> {
    let (inputs, next_states) = generate_7cell_data();

    let in_dim = 7; // each neighborhood has 3 cells
    let hidden_dim = 4; // hidden dimension for MLP
    let out_dim = 1; // output dimension (next state)

    let device = Device::Cpu;

    let xs = Tensor::from_vec(inputs.concat(), (inputs.len(), in_dim), &device)?;

    let ys = Tensor::from_vec(next_states, (inputs.len(), out_dim), &device)?;

    let trained_vs = train_transformer(in_dim, hidden_dim, out_dim, xs, ys, 1000, device)?;

    let st = Storage::new();
    let mut solver = Solver::new(&st, Z3Binary::new("/usr/bin/z3")?)?; // Note: make sure Z3 is installed at your machine and the path is correct

    let x = Int::new_const(&st, "x");
    let y = Int::new_const(&st, "y");

    solver.assert(x._eq(y + 25))?;
    solver.assert(x._eq(204))?;

    // Check for validity
    match solver.check_sat_with_model()? {
        SatResultWithModel::Sat(model) => {
            // Since it is valid, we can extract the possible values of the
            // variables using a model
            println!("x = {}", model.eval(x).unwrap());
            println!("y = {}", model.eval(y).unwrap());
        }
        SatResultWithModel::Unsat => println!("No valid solutions found!"),
        SatResultWithModel::Unknown => println!("Satisfaction remains unknown..."),
    }

    Ok(())
}
