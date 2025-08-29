use anyhow::anyhow;
use smtlib::backend::z3_binary::Z3Binary;
use smtlib::prelude::*;
use smtlib::terms::Const;
use smtlib::{Real, Solver, Storage};

fn generate_variables(n: usize, st: &Storage) -> Vec<Const<Real>> {
    (0..n)
        .map(|i| Real::new_const(st, format!("x{}", i).as_str()))
        .collect()
}

// Outputs the auxiliary variables needed for the projection constraints
fn generate_projection_constraints<'a>(
    variables: &'a Vec<Const<Real<'a>>>,
    weights: Vec<Vec<f64>>, // the size is variables.len() x variables.len()
    st: &'a Storage,
    solver: &'a mut Solver<'a, Z3Binary>,
    name: &str,
) -> anyhow::Result<Vec<Const<'a, Real<'a>>>> {
    let dim = variables.len();
    (0..dim)
        .map(|i| {
            let aux_var = Real::new_const(st, format!("{}{}", name, i).as_str());
            match solver.assert(
                aux_var._eq(
                    variables
                        .iter()
                        .enumerate()
                        .skip(1) // skipping as we start from weights[i][0] * variables[0]
                        .fold(variables[0] * weights[i][0], |mut acc, (j, v)| {
                            acc + (*v * weights[i][j])
                        }),
                ),
            ) {
                Ok(_) => Ok(aux_var),
                Err(e) => Err(anyhow!("Issue asserting projection constraint: {}", e)),
            }
        })
        .collect::<anyhow::Result<Vec<Const<'a, Real<'a>>>>>()
}

// We are approximatinng the softmax with a >= 0 and sum(a) = 1.0 constraint
fn generate_q_k_dot_product_constraints_with_axioms<'a>(
    q_vars: &'a Vec<Const<Real<'a>>>,
    k_vars: &'a Vec<Const<Real<'a>>>,
    st: &'a Storage,
    solver: &'a mut Solver<'a, Z3Binary>,
) -> anyhow::Result<Vec<Const<'a, Real<'a>>>> {
    let dim = q_vars.len();
    let scaling_constant = 1.0 / (dim as f64).sqrt();
    let pos_axiom = Real::new_const(st, "pos_axiom");
    solver.assert(pos_axiom._eq(0.0))?;

    let mut aux_vars = Vec::with_capacity(dim * dim);

    for i in 0..dim {
        let mut row_vars = Vec::with_capacity(dim);
        for j in 0..dim {
            let aux_var = Real::new_const(st, format!("qk_{}_{}", i, j).as_str());
            solver.assert(aux_var._eq(q_vars[i] * k_vars[j] * scaling_constant))?;
            solver.assert(aux_var.ge(pos_axiom))?;
            row_vars.push(aux_var);
        }

        let sum_var = row_vars
            .iter()
            .map(|v| Real::from(*v))
            .reduce(|acc, v| acc + v)
            .ok_or(anyhow!("Issue creating sum for qk row"))?;
        solver.assert(sum_var._eq(1.0))?;
        aux_vars.extend(row_vars);
    }

    Ok(aux_vars)
}
