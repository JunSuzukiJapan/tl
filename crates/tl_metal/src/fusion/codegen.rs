//! MSL コード生成 — 融合グループから MSL カーネルソースを生成
//!
//! 操作チェーンを解析し、1つの MSL カーネル関数を生成する。

use tl_backend::fusion::ElementWiseOp;

/// 操作ノード（コード生成用）
#[derive(Debug, Clone)]
pub struct CodeGenNode {
    pub op: ElementWiseOp,
    /// 入力ノード ID (OpGraph 内の ID)
    pub inputs: Vec<usize>,
}

/// MSL カーネルソースを生成する
pub struct MslCodeGen;

impl MslCodeGen {
    /// 操作チェーンから MSL カーネルを生成
    ///
    /// `leaf_count`: 外部入力テンソルの数
    /// `nodes`: 操作ノードのリスト（トポロジカル順）
    /// `output_node`: 出力ノードの ID
    /// `kernel_name`: カーネル関数名
    pub fn generate(
        leaf_count: usize,
        nodes: &[CodeGenNode],
        output_node: usize,
        kernel_name: &str,
    ) -> String {
        let mut src = String::new();
        src.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

        // カーネルシグネチャ
        src.push_str(&format!("kernel void {}(\n", kernel_name));

        // 入力バッファ
        for i in 0..leaf_count {
            src.push_str(&format!(
                "    device const float* in{} [[buffer({})]],\n", i, i
            ));
        }

        // 出力バッファ
        src.push_str(&format!(
            "    device float* out [[buffer({})]],\n", leaf_count
        ));

        // count パラメータ
        src.push_str(&format!(
            "    constant uint& count [[buffer({})]],\n", leaf_count + 1
        ));

        src.push_str("    uint id [[thread_position_in_grid]]\n");
        src.push_str(") {\n");
        src.push_str("    if (id >= count) return;\n\n");

        // リーフ入力の読み込み
        for i in 0..leaf_count {
            src.push_str(&format!("    float v{} = in{}[id];\n", i, i));
        }
        src.push('\n');

        // 各ノードの計算式
        for (idx, node) in nodes.iter().enumerate() {
            let node_id = leaf_count + idx;
            let input_names: Vec<String> = node.inputs.iter()
                .map(|&i| format!("v{}", i))
                .collect();
            let expr = node.op.to_msl_expr(&input_names);
            src.push_str(&format!("    float v{} = {};\n", node_id, expr));
        }

        // 出力書き込み
        src.push_str(&format!("\n    out[id] = v{};\n", output_node));
        src.push_str("}\n");

        src
    }
}
