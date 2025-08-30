use anyhow::anyhow;
use easy_smt::{Context, ContextBuilder, SExpr};
use smtlib::backend::z3_binary::Z3Binary;
use smtlib::prelude::*;
use smtlib::terms::Const;
use smtlib::{Real, Solver, Storage};

fn generate_variables(
    n: usize,
    mut ctx: Context,
    sort: SExpr,
    non_zero: bool, // Enforce at least one variable to be non-zero
) -> anyhow::Result<(Vec<SExpr>, Context)> {
    let vars = (0..n)
        .map(|i| {
            let x = ctx
                .declare_const(format!("x{}", i).as_str(), sort)
                .map_err(|e| anyhow!("Issue declaring variable: {}", e))?;
            // Ensure variable is boolean (0.0 or 1.0)
            ctx.assert(ctx.or(ctx.eq(x, ctx.decimal(0.0)), ctx.eq(x, ctx.decimal(1.0))))
                .map_err(|e| anyhow!("Issue declaring variable: {}", e))?;
            Ok(x)
        })
        .collect::<anyhow::Result<Vec<SExpr>>>()?;
    if non_zero {
        let non_zero_constraint = ctx.or_many(
            vars.iter()
                .map(|var| ctx.not(ctx.eq(*var, ctx.decimal(0.0)))),
        );
        // ctx.assert(non_zero_constraint)
        //     .map_err(|e| anyhow!("Issue declaring non-zero constraint: {}", e))?;
        let sum = vars
            .iter()
            .fold(ctx.decimal(0.0), |acc, v| ctx.plus(acc, *v));
        ctx.assert(ctx.gt(sum, ctx.decimal(1.0)))
            .map_err(|e| anyhow!("Issue declaring non-zero constraint: {}", e))?;
    }
    Ok((vars, ctx))
}

// Outputs the linear constraints as consts
// The output size is weight_rows x var_cols
fn generate_linear_constraints(
    weights: Vec<Vec<f64>>,
    weight_rows: usize,
    weight_cols: usize,
    bias: Option<Vec<f64>>,
    variables: &Vec<&Vec<SExpr>>,
    var_rows: usize,
    var_cols: usize,
    mut ctx: Context,
    sort: SExpr,
    name: &str,
) -> anyhow::Result<(Vec<Vec<SExpr>>, Context)> {
    let mut const_rows = vec![];
    for i in 0..weight_rows {
        let mut row = vec![];
        for j in 0..var_cols {
            let aux_var = ctx
                .declare_const(format!("{}{}_{}", name, i, j).as_str(), sort)
                .map_err(|e| anyhow!("Issue declaring linear constraint const"))?;
            let mut result = ctx.decimal(0.0);
            for k in 0..weight_cols {
                for l in 0..var_rows {
                    result = ctx.plus(
                        result,
                        ctx.times(ctx.decimal(weights[i][k]), variables[l][j]),
                    );
                }
            }
            if let Some(bias) = &bias {
                result = ctx.plus(result, ctx.decimal(bias[i]));
            }
            ctx.assert(ctx.eq(aux_var, result))?;
            row.push(aux_var);
        }
        const_rows.push(row);
    }

    Ok((const_rows, ctx))
}

fn generate_relu_constraints<'a>(
    input_vars: &'a Vec<Const<Real<'a>>>,
    st: &'a Storage,
    mut solver: Solver<'a, Z3Binary>,
) -> anyhow::Result<Solver<'a, Z3Binary>> {
    let dim = input_vars.len();
    let pos_relu_axiom = Real::new_const(st, "pos_relu_axiom");
    solver.assert(pos_relu_axiom._eq(0.0))?;
    for var in input_vars {
        solver.assert(var.ge(pos_relu_axiom).ite(var._eq(*var), var._eq(0.0)))?;
    }
    Ok(solver)
}

// We are approximating the softmax with a >= 0 and sum(a) = 1.0 constraint
fn generate_q_k_dot_product_constraints_with_axioms(
    q_vars: &Vec<Vec<SExpr>>, // we transpose Q in our case -> we need to swap indices
    k_vars: &Vec<Vec<SExpr>>,
    dim_k: usize,
    seq_len: usize, // Output is the square matix N x N (seq_len x seq_len)
    mut ctx: Context,
    sort: SExpr,
) -> anyhow::Result<(Vec<Vec<SExpr>>, Context)> {
    let scaling_constant = ctx.decimal(1.0 / (dim_k as f64).sqrt());

    let mut const_rows = vec![];
    for i in 0..seq_len {
        let mut row = vec![];
        for j in 0..seq_len {
            let aux_var = ctx
                .declare_const(format!("qk{}_{}", i, j).as_str(), sort)
                .map_err(|e| anyhow!("Issue declaring linear constraint const"))?;
            let mut result = ctx.decimal(0.0);
            for k in 0..dim_k {
                for l in 0..dim_k {
                    result = ctx.plus(result, ctx.times(q_vars[k][i], k_vars[l][j]));
                }
            }
            result = ctx.times(result, scaling_constant);
            ctx.assert(ctx.eq(aux_var, result))?;
            // NOTE: we are asserting thresholds instead of enforcing softmax
            //ctx.assert(ctx.gte(aux_var, ctx.decimal(0.0)))?;
            //ctx.assert(ctx.lte(aux_var, ctx.decimal(1.0)))?;

            // NOTE: I have removed the axiom that the sum of each row is 1.0 for now
            // This is because it is making the problem infeasible in some cases
            row.push(aux_var);
        }
        const_rows.push(row);
    }

    Ok((const_rows, ctx))
}

fn generate_qkv_constraints(
    v_vars: &Vec<Vec<SExpr>>,
    qk_vars: &Vec<Vec<SExpr>>,
    dim_k: usize,
    seq_len: usize,
    mut ctx: Context,
    sort: SExpr,
) -> anyhow::Result<(Vec<Vec<SExpr>>, Context)> {
    let mut const_rows = vec![];
    for i in 0..dim_k {
        let mut row = vec![];
        for j in 0..seq_len {
            let aux_var = ctx
                .declare_const(format!("qkv{}_{}", i, j).as_str(), sort)
                .map_err(|e| anyhow!("Issue declaring linear constraint const"))?;
            let mut result = ctx.decimal(0.0);
            for k in 0..seq_len {
                for l in 0..seq_len {
                    result = ctx.plus(result, ctx.times(v_vars[i][k], qk_vars[l][j]));
                }
            }
            ctx.assert(ctx.eq(aux_var, result))?;
            row.push(aux_var);
        }
        const_rows.push(row);
    }

    Ok((const_rows, ctx))
}

pub fn verify_transformer_block<'a>(
    non_zero: bool, // enforce at least one input variable to be non-zero
    in_dim: usize,
    seq_len: usize,
    d_model: usize,
    hidden_dim: usize,
    //solver: &'a mut Solver<'a, Z3Binary>,
    weights_emb: Vec<Vec<f64>>,
    weights_q: Vec<Vec<f64>>,
    weights_k: Vec<Vec<f64>>,
    weights_v: Vec<Vec<f64>>,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
) -> anyhow::Result<()> {
    let mut ctx = ContextBuilder::new().with_z3_defaults().build()?;

    let real = ctx.real_sort();

    let (variables, ctx) = generate_variables(seq_len, ctx, real, non_zero)?;

    let (emb_constraints, ctx) = generate_linear_constraints(
        weights_emb,
        d_model,
        in_dim,
        None,
        &vec![&variables],
        in_dim,
        seq_len,
        ctx,
        real,
        "emb_",
    )?;

    println!("Emb rows: {}", emb_constraints.len());
    println!("Emb cols: {}", emb_constraints[0].len());

    let (q_constraints, ctx) = generate_linear_constraints(
        weights_q,
        d_model,
        d_model,
        None,
        &emb_constraints.iter().collect(),
        d_model,
        seq_len,
        ctx,
        real,
        "q_",
    )?;
    println!("Q rows: {}", q_constraints.len());
    println!("Q cols: {}", q_constraints[0].len());

    let (k_constraints, ctx) = generate_linear_constraints(
        weights_k,
        d_model,
        d_model,
        None,
        &emb_constraints.iter().collect(),
        d_model,
        seq_len,
        ctx,
        real,
        "k_",
    )?;
    println!("K rows: {}", k_constraints.len());
    println!("K cols: {}", k_constraints[0].len());

    let (v_constraints, mut ctx) = generate_linear_constraints(
        weights_v,
        d_model,
        d_model,
        None,
        &emb_constraints.iter().collect(),
        d_model,
        seq_len,
        ctx,
        real,
        "v_",
    )?;

    println!("V rows: {}", v_constraints.len());
    println!("V cols: {}", v_constraints[0].len());

    let (qk_constraints, ctx) = generate_q_k_dot_product_constraints_with_axioms(
        &q_constraints,
        &k_constraints,
        d_model,
        seq_len,
        ctx,
        real,
    )?;

    println!("QK rows: {}", qk_constraints.len());
    println!("QK cols: {}", qk_constraints[0].len());

    let (qkv_constraints, mut ctx) =
        generate_qkv_constraints(&v_constraints, &qk_constraints, d_model, seq_len, ctx, real)?;

    println!("QKV rows: {}", qkv_constraints.len());
    println!("QKV cols: {}", qkv_constraints[0].len());

    let check_result = ctx.check();

    println!("Check result: {:?}", check_result);

    // Check whether the assertions are satisfiable. They should be in this example.
    //assert_eq!(ctx.check()?, Response::Sat);

    // Print the solution!
    let solution = ctx.get_value(variables)?;
    for (variable, value) in solution {
        println!("{} = {}", ctx.display(variable), ctx.display(value));
    }
    let solution = ctx.get_value(emb_constraints.into_iter().flatten().collect())?;
    for (variable, value) in solution {
        println!("{} = {}", ctx.display(variable), ctx.display(value));
    }
    let solution = ctx.get_value(q_constraints.into_iter().flatten().collect())?;
    for (variable, value) in solution {
        println!("{} = {}", ctx.display(variable), ctx.display(value));
    }
    let solution = ctx.get_value(k_constraints.into_iter().flatten().collect())?;
    for (variable, value) in solution {
        println!("{} = {}", ctx.display(variable), ctx.display(value));
    }
    let solution = ctx.get_value(v_constraints.into_iter().flatten().collect())?;
    for (variable, value) in solution {
        println!("{} = {}", ctx.display(variable), ctx.display(value));
    }
    let solution = ctx.get_value(qk_constraints.into_iter().flatten().collect())?; // qk_constraints.into_iter().flatten().collect()
    for (variable, value) in solution {
        println!("{} = {}", ctx.display(variable), ctx.display(value));
    }

    let solution = ctx.get_value(qkv_constraints.into_iter().flatten().collect())?; // qk_constraints.into_iter().flatten().collect()
    for (variable, value) in solution {
        println!("{} = {}", ctx.display(variable), ctx.display(value));
    }

    // let (q_constraints, mut ctx) =
    //     generate_linear_constraints(&variables, weights_q, None, ctx, real, "q_")?;

    // let (k_constraints, ctx) =
    //     generate_linear_constraints(&variables, weights_k, None, ctx, real, "k_")?;
    // let (v_constraints, ctx) =
    //     generate_linear_constraints(&variables, weights_v, None, ctx, real, "v_")?;
    //
    // let (qk_constraints, mut ctx) = generate_q_k_dot_product_constraints_with_axioms(
    //     &q_constraints,
    //     &k_constraints,
    //     ctx,
    //     real,
    // )?;

    // let (qkv_constraints, solver) =
    //     generate_qkv_constraints(&qk_constraints, &v_constraints, &st, solver)?;
    //
    // let (w1_constraints, solver) =
    //     generate_linear_constraints(&qkv_constraints, w1, Some(b1), &st, solver, "w1_")?;
    // let solver = generate_relu_constraints(&w1_constraints, &st, solver)?;
    // let (w2_constraints, solver) =
    //     generate_linear_constraints(&w1_constraints, w2, Some(b2), &st, solver, "w2_")?;

    Ok(())
}
