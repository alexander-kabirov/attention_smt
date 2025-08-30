use candle_core::{DType, Device, Module, Tensor};
use candle_nn::init::DEFAULT_KAIMING_NORMAL;
use candle_nn::loss::mse;
use candle_nn::{ops, Init, Optimizer, ParamsAdamW, VarBuilder, VarMap};

// A simple Linear layer implementation
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
        let x = match *xs.dims() {
            [b1, b2, m, k] => {
                if xs.is_contiguous() {
                    let w = self.weight.t()?;
                    xs.reshape((b1 * b2 * m, k))?
                        .matmul(&w)?
                        .reshape((b1, b2, m, ()))?
                } else {
                    let w = self.weight.broadcast_left((b1, b2))?.t()?;
                    xs.matmul(&w)?
                }
            }
            [bsize, m, k] => {
                if xs.is_contiguous() {
                    let w = self.weight.t()?;
                    xs.reshape((bsize * m, k))?
                        .matmul(&w)?
                        .reshape((bsize, m, ()))?
                } else {
                    let w = self.weight.broadcast_left(bsize)?.t()?;
                    xs.matmul(&w)?
                }
            }
            _ => {
                let w = self.weight.t()?;
                xs.matmul(&w)?
            }
        };
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

// A naive single-head attention mechanism
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

// A simple 2-layer MLP with ReLU activation
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

// A naive Transformer block with a single attention head and a simple MLP
pub struct NaiveTransformerBlock {
    d_model: usize,
    seq_len: usize,
    emb: Linear,
    attn: NaiveAttentionHead,
    mlp: NaiveMLP,
}

impl NaiveTransformerBlock {
    fn new(
        in_dim: usize,
        seq_len: usize,
        d_model: usize,
        hidden_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
    ) -> anyhow::Result<Self> {
        let emb = linear_no_bias(in_dim, d_model, vb.pp("emb"))?; // input embedding layer
        let attn = NaiveAttentionHead::new(d_model, d_model, vb.pp("attn"))?; // d_model = d_k
        let mlp = NaiveMLP::new(d_model * seq_len, hidden_dim, out_dim, vb.pp("mlp"))?;
        Ok(Self {
            d_model,
            seq_len,
            emb,
            attn,
            mlp,
        })
    }
}

impl Module for NaiveTransformerBlock {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let emb = self.emb.forward(xs)?; // input embedding
        let attn_out = self.attn.forward(&emb)?; // attention output
        let attn_res = (emb + attn_out)?.reshape(((), self.d_model * self.seq_len))?; // residual connection
        self.mlp.forward(&attn_res) // Notice the oversimplification here: no residual connection after MLP
    }
}

// Note the MSE loss instead of Cross-Entropy which would make more sense for a classification task, but the former plays better with linear SMT
pub fn train_transformer<'a>(
    in_dim: usize,
    seq_len: usize,
    d_model: usize,
    hidden_dim: usize,
    out_dim: usize,
    xs: Tensor,
    ys: Tensor,
    epochs: usize,
    device: Device,
) -> anyhow::Result<VarBuilder<'a>> {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let transformer_block = NaiveTransformerBlock::new(
        in_dim,
        seq_len,
        d_model,
        hidden_dim,
        out_dim,
        vs.pp("transformer"),
    )?;

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

pub fn get_trained_weights(
    vs: &VarBuilder,
    in_dim: usize,
    seq_len: usize,
    d_model: usize,
    hidden_dim: usize,
    out_dim: usize,
) -> anyhow::Result<(
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<f64>,
)> {
    let weights_emb = vs
        .get((d_model, in_dim), "transformer.emb.weight")?
        .to_vec2::<f32>()?
        .into_iter()
        .map(|vec| vec.into_iter().map(|x| x as f64).collect())
        .collect();
    let weights_q = vs
        .get((d_model, d_model), "transformer.attn.query.weight")?
        .to_vec2::<f32>()?
        .into_iter()
        .map(|vec| vec.into_iter().map(|x| x as f64).collect())
        .collect();
    let weights_k = vs
        .get((d_model, d_model), "transformer.attn.key.weight")?
        .to_vec2::<f32>()?
        .into_iter()
        .map(|vec| vec.into_iter().map(|x| x as f64).collect())
        .collect();
    let weights_v = vs
        .get((d_model, d_model), "transformer.attn.value.weight")?
        .to_vec2::<f32>()?
        .into_iter()
        .map(|vec| vec.into_iter().map(|x| x as f64).collect())
        .collect();
    let w1 = vs
        .get((hidden_dim, d_model * seq_len), "transformer.mlp.w1.weight")?
        .to_vec2::<f32>()?
        .into_iter()
        .map(|vec| vec.into_iter().map(|x| x as f64).collect())
        .collect();
    let b1 = vs
        .get(hidden_dim, "transformer.mlp.w1.bias")?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|x| x as f64)
        .collect();
    let w2 = vs
        .get((out_dim, hidden_dim), "transformer.mlp.w2.weight")?
        .to_vec2::<f32>()?
        .into_iter()
        .map(|vec| vec.into_iter().map(|x| x as f64).collect())
        .collect();
    let b2 = vs
        .get(out_dim, "transformer.mlp.w2.bias")?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|x| x as f64)
        .collect();
    Ok((weights_emb, weights_q, weights_k, weights_v, w1, b1, w2, b2))
}
