use anyhow::anyhow;
use easy_smt::{Context, ContextBuilder, Response, SExpr};

// Generates n Real variables and adds them to the context, we also enforce them to be boolean (0.0 or 1.0)
// If non_zero is true, we enforce at least one variable to be non-zero
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

// Outputs the linear transformation constraints as consts
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
    variables: &Vec<&Vec<SExpr>>,
    var_rows: usize,
    var_cols: usize,
    mut ctx: Context,
    sort: SExpr,
    name: &str,
) -> anyhow::Result<(Vec<Vec<SExpr>>, Context)> {
    let mut const_rows = vec![];
    for i in 0..var_rows {
        let mut row = vec![];
        for j in 0..var_cols {
            let aux_var = ctx
                .declare_const(format!("{}{}_{}", name, i, j).as_str(), sort)
                .map_err(|e| anyhow!("Issue declaring linear constraint const"))?;
            let relu = ctx.ite(
                ctx.gte(variables[i][j], ctx.decimal(0.0)),
                variables[i][j],
                ctx.decimal(0.0),
            );
            ctx.assert(ctx.eq(aux_var, relu))?;
            row.push(aux_var);
        }
        const_rows.push(row);
    }
    Ok((const_rows, ctx))
}

// We may try approximating the softmax with a >= 0 and sum(a) = 1.0 constraint
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

fn residual_connection_constraints(
    emb_vars: &Vec<Vec<SExpr>>,
    qkv_vars: &Vec<Vec<SExpr>>,
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
                .declare_const(format!("res{}_{}", i, j).as_str(), sort)
                .map_err(|e| anyhow!("Issue declaring linear constraint const"))?;
            let result = ctx.plus(emb_vars[i][j], qkv_vars[i][j]);
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

    let (qk_constraints, ctx) = generate_q_k_dot_product_constraints_with_axioms(
        &q_constraints,
        &k_constraints,
        d_model,
        seq_len,
        ctx,
        real,
    )?;

    let (qkv_constraints, ctx) =
        generate_qkv_constraints(&v_constraints, &qk_constraints, d_model, seq_len, ctx, real)?;

    let (res_constraints, ctx) = residual_connection_constraints(
        &emb_constraints,
        &qkv_constraints,
        d_model,
        seq_len,
        ctx,
        real,
    )?;

    let (w1_constraints, ctx) = generate_linear_constraints(
        w1,
        hidden_dim,
        d_model * seq_len,
        Some(b1),
        &res_constraints
            .iter()
            .flatten()
            .map(|constraint| vec![*constraint])
            .collect::<Vec<Vec<SExpr>>>()
            .iter()
            .collect(), // we unroll the matrix into a vector
        d_model * seq_len,
        1,
        ctx,
        real,
        "w1_",
    )?;

    let (relu_constraints, ctx) = generate_relu_constraints(
        &w1_constraints.iter().collect(),
        hidden_dim,
        1,
        ctx,
        real,
        "w1_relu_",
    )?;

    let (w2_constraints, mut ctx) = generate_linear_constraints(
        w2,
        1, // output dim is 1
        hidden_dim,
        Some(b2),
        &relu_constraints.iter().collect(),
        hidden_dim,
        1,
        ctx,
        real,
        "w2_",
    )?;

    // Add sample assertions:
    // 1. Counterfactuals: An input that should produce 1 outputs a 0 and vice versa (threshold-based)
    // GOAL: we need to see an UNSAT result (i.e., no counterexample exists)
    // Different training runs to provide different results, if we found some counterexamples we show them
    // Next we could add those counterexamples as additional constraints to get more examples
    // We can verify the attention patterns too in a similar way

    let boolean = ctx.bool_sort();
    let x0_bool = ctx.declare_const("x0_bool", boolean)?;
    ctx.assert(ctx.eq(x0_bool, ctx.eq(variables[0], ctx.decimal(1.0))))?;
    let x2_bool = ctx.declare_const("x2_bool", boolean)?;
    ctx.assert(ctx.eq(x2_bool, ctx.eq(variables[2], ctx.decimal(1.0))))?;
    let x4_bool = ctx.declare_const("x4_bool", boolean)?;
    ctx.assert(ctx.eq(x4_bool, ctx.eq(variables[4], ctx.decimal(1.0))))?;
    ctx.assert(ctx.or(
        ctx.and(
            ctx.eq(ctx.xor(ctx.xor(x0_bool, x2_bool), x4_bool), ctx.true_()),
            ctx.lt(w2_constraints[0][0], ctx.decimal(0.0)),
        ),
        ctx.and(
            ctx.eq(ctx.xor(ctx.xor(x0_bool, x2_bool), x4_bool), ctx.false_()),
            ctx.gt(w2_constraints[0][0], ctx.decimal(0.0)),
        ),
    ))?;

    let check_result = ctx.check();

    match check_result {
        Ok(result) => match result {
            Response::Sat => {
                println!("SAT, Here is a counterexample:");
                let solution = ctx.get_value(variables)?;
                for (variable, value) in solution {
                    println!("{} = {}", ctx.display(variable), ctx.display(value));
                }

                let solution = ctx.get_value(vec![w2_constraints[0][0]])?;
                for (variable, value) in solution {
                    println!("{} = {}", ctx.display(variable), ctx.display(value));
                }
            }
            Response::Unsat => {
                println!("UNSAT (No counterexamples exists)");
            }
            Response::Unknown => {
                println!("UNKNOWN result from SMT solver");
            }
        },
        Err(e) => {
            println!("Error during check: {}", e);
            return Err(anyhow!("Error during SMT check"));
        }
    }
    Ok(())
}
