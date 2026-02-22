//! GpuFusion — 自動カーネル融合トレイト
//!
//! 複数の element-wise 操作を遅延実行し、
//! ランタイムで融合カーネルを生成・実行する。

use std::fmt;

/// 融合可能な element-wise 操作
#[derive(Debug, Clone, PartialEq)]
pub enum ElementWiseOp {
    // 単項
    Neg, Abs, Exp, Log, Sqrt, Sin, Cos, Tanh,
    Sigmoid, Relu, Gelu, Silu,
    // 二項（第2入力が別テンソル）
    Add, Sub, Mul, Div, Pow,
    // スカラー演算
    AddScalar(f32),
    MulScalar(f32),
}

impl ElementWiseOp {
    /// 入力テンソル数（リーフ入力を除く）
    pub fn num_inputs(&self) -> usize {
        match self {
            // 二項演算: 2入力
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Pow => 2,
            // それ以外: 1入力
            _ => 1,
        }
    }

    /// MSL コードの式を生成
    pub fn to_msl_expr(&self, inputs: &[String]) -> String {
        match self {
            Self::Neg => format!("(-{})", inputs[0]),
            Self::Abs => format!("abs({})", inputs[0]),
            Self::Exp => format!("exp({})", inputs[0]),
            Self::Log => format!("log({})", inputs[0]),
            Self::Sqrt => format!("sqrt({})", inputs[0]),
            Self::Sin => format!("sin({})", inputs[0]),
            Self::Cos => format!("cos({})", inputs[0]),
            Self::Tanh => format!("tanh({})", inputs[0]),
            Self::Sigmoid => format!("(1.0f / (1.0f + exp(-{})))", inputs[0]),
            Self::Relu => format!("max({}, 0.0f)", inputs[0]),
            Self::Gelu => {
                let x = &inputs[0];
                format!("({x} * 0.5f * (1.0f + tanh(0.7978845608f * ({x} + 0.044715f * {x} * {x} * {x}))))")
            }
            Self::Silu => {
                let x = &inputs[0];
                format!("({x} / (1.0f + exp(-{x})))")
            }
            Self::Add => format!("({} + {})", inputs[0], inputs[1]),
            Self::Sub => format!("({} - {})", inputs[0], inputs[1]),
            Self::Mul => format!("({} * {})", inputs[0], inputs[1]),
            Self::Div => format!("({} / {})", inputs[0], inputs[1]),
            Self::Pow => format!("pow({}, {})", inputs[0], inputs[1]),
            Self::AddScalar(s) => format!("({} + {:.8}f)", inputs[0], s),
            Self::MulScalar(s) => format!("({} * {:.8}f)", inputs[0], s),
        }
    }

    /// キャッシュキー用の文字列表現
    pub fn cache_key(&self) -> String {
        match self {
            Self::Neg => "neg".into(),
            Self::Abs => "abs".into(),
            Self::Exp => "exp".into(),
            Self::Log => "log".into(),
            Self::Sqrt => "sqrt".into(),
            Self::Sin => "sin".into(),
            Self::Cos => "cos".into(),
            Self::Tanh => "tanh".into(),
            Self::Sigmoid => "sigmoid".into(),
            Self::Relu => "relu".into(),
            Self::Gelu => "gelu".into(),
            Self::Silu => "silu".into(),
            Self::Add => "add".into(),
            Self::Sub => "sub".into(),
            Self::Mul => "mul".into(),
            Self::Div => "div".into(),
            Self::Pow => "pow".into(),
            Self::AddScalar(s) => format!("adds_{}", s.to_bits()),
            Self::MulScalar(s) => format!("muls_{}", s.to_bits()),
        }
    }
}

impl fmt::Display for ElementWiseOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.cache_key())
    }
}
