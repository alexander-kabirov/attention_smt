use candle_core::{DType, Device, Module, Tensor};
use candle_nn::init::DEFAULT_KAIMING_NORMAL;
use candle_nn::loss::mse;
use candle_nn::{ops, Init, Optimizer, ParamsAdamW, VarBuilder, VarMap};

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

// Create or initialize a new linear layer wit biases.
pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> anyhow::Result<Linear> {
    let init_ws = DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}

// Create or initialize a new linear layer without biases.
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> anyhow::Result<Linear> {
    let init_ws = DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(Linear::new(ws, None))
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let w = self.weight.t()?;
        let x = xs.matmul(&w)?; // we are handling only the simples 2D case here
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub struct NaiveAttentionHead {
    in_dim: usize, // in our toy sample we are using a single head hence in_dim = out_dim
    scaling_const: Tensor,
    w_key: Linear,
    w_query: Linear,
    w_value: Linear,
}

impl NaiveAttentionHead {
    fn new(in_dim: usize, hidden_dim: usize, vb: VarBuilder) -> anyhow::Result<Self> {
        let w_key = linear_no_bias(in_dim, hidden_dim, vb.pp("key"))?;
        let w_query = linear_no_bias(in_dim, hidden_dim, vb.pp("query"))?;
        let w_value = linear_no_bias(in_dim, hidden_dim, vb.pp("value"))?;
        Ok(Self {
            in_dim,
            scaling_const: Tensor::from_vec(
                vec![1.0 / (hidden_dim as f32).sqrt()],
                (),
                &vb.device(),
            )?,
            w_key,
            w_query,
            w_value,
        })
    }
}

impl Module for NaiveAttentionHead {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let keys = self.w_key.forward(xs)?; // key projection
        let queries = self.w_query.forward(xs)?; // query projection
        let values = self.w_value.forward(xs)?; // value projection
        let scores = queries
            .matmul(&keys.t()?)?
            .broadcast_div(&self.scaling_const)?; // scaled dot-product
        ops::softmax(&scores, 1)?.matmul(&values) // attention output, hardcoding dim=1
    }
}

pub struct NaiveMLP {
    w1: Linear,
    w2: Linear,
}

impl NaiveMLP {
    fn new(
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
    ) -> anyhow::Result<Self> {
        let w1 = linear(in_dim, hidden_dim, vb.pp("w1"))?;
        let w2 = linear(hidden_dim, out_dim, vb.pp("w2"))?;
        Ok(Self { w1, w2 })
    }
}

impl Module for NaiveMLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.w1.forward(xs)?.relu()?; // first layer with ReLU activation
        self.w2.forward(&h) // second layer
    }
}

pub struct NaiveTransformerBlock {
    attn: NaiveAttentionHead,
    mlp: NaiveMLP,
}

impl NaiveTransformerBlock {
    fn new(
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
    ) -> anyhow::Result<Self> {
        let attn = NaiveAttentionHead::new(in_dim, in_dim, vb.pp("attn"))?; // Notice that since we are using a single head, hidden_dim = in_dim
        let mlp = NaiveMLP::new(in_dim, hidden_dim, out_dim, vb.pp("mlp"))?;
        Ok(Self { attn, mlp })
    }
}

impl Module for NaiveTransformerBlock {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let attn_out = self.attn.forward(xs)?; // attention output
        let attn_res = (xs + attn_out)?; // residual connection
        self.mlp.forward(&attn_res) // Notice the oversimplification here: no residual connection after MLP
    }
}

// Note the MSE loss instead of Cross-Entropy which would make more sense for a classification task, but the former plays better with linear SMT
fn train_transformer<'a>(
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
    xs: Tensor,
    ys: Tensor,
    epochs: usize,
    device: Device,
) -> anyhow::Result<VarBuilder<'a>> {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let transformer_block =
        NaiveTransformerBlock::new(in_dim, hidden_dim, out_dim, vs.pp("transformer"))?;

    let mut adam_params = ParamsAdamW::default();
    adam_params.lr = 0.01;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), adam_params)?;

    for epoch in 0..epochs {
        let pred = transformer_block.forward(&xs)?;

        let loss = mse(&pred, &ys)?;

        optimizer.backward_step(&loss)?;

        if epoch % 100 == 0 {
            println!("Prediction at epoch {}: {}", epoch, pred);
            println!("Epoch {}: loss = {}", epoch, loss);
        }
    }

    Ok(vs)
}

fn main() -> anyhow::Result<()> {
    // Define toy 1D CA dataset (Rule 110)
    let neighborhoods: Vec<[f32; 3]> = vec![
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ];
    let next_states: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]; // Rule 110

    let in_dim = 3; // each neighborhood has 3 cells
    let hidden_dim = 4; // hidden dimension for MLP
    let out_dim = 1; // output dimension (next state)

    let device = Device::Cpu;

    let xs = Tensor::from_vec(
        neighborhoods.concat(),
        (neighborhoods.len(), in_dim),
        &device,
    )?;

    let ys = Tensor::from_vec(next_states, (neighborhoods.len(), out_dim), &device)?;

    let trained_vs = train_transformer(in_dim, hidden_dim, out_dim, xs, ys, 1000, device)?;

    Ok(())
}
