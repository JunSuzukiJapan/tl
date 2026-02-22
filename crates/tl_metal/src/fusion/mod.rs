//! 自動カーネル融合 — Metal バックエンド
//!
//! `LazyTensor` で操作を DAG として記録し、
//! `materialize()` で融合カーネルを生成・実行する。

pub mod codegen;
pub mod cache;

use crate::{MetalTensor, DType};
use crate::device::get_device;
use crate::command_stream::stream_encode;
use crate::shaders::compute_thread_groups;
use codegen::{MslCodeGen, CodeGenNode};
use cache::get_cache;
use tl_backend::fusion::ElementWiseOp;
use std::sync::Arc;

// ============================================================
// OpGraph: 操作 DAG
// ============================================================

/// DAG ノードの種類
enum NodeKind {
    /// リーフ: 実テンソルへの参照
    Leaf(Arc<MetalTensor>),
    /// 操作ノード
    Op {
        op: ElementWiseOp,
        inputs: Vec<usize>, // ノード ID
    },
}

/// DAG ノード
struct OpNode {
    kind: NodeKind,
    shape: Vec<usize>,
}

/// 操作 DAG
struct OpGraph {
    nodes: Vec<OpNode>,
}

impl OpGraph {
    fn new() -> Self {
        OpGraph { nodes: Vec::new() }
    }

    /// リーフノードを追加
    fn add_leaf(&mut self, tensor: Arc<MetalTensor>) -> usize {
        let shape = tensor.shape().to_vec();
        let id = self.nodes.len();
        self.nodes.push(OpNode {
            kind: NodeKind::Leaf(tensor),
            shape,
        });
        id
    }

    /// 操作ノードを追加
    fn add_op(&mut self, op: ElementWiseOp, inputs: Vec<usize>) -> usize {
        let shape = self.nodes[inputs[0]].shape.clone();
        let id = self.nodes.len();
        self.nodes.push(OpNode {
            kind: NodeKind::Op { op, inputs },
            shape,
        });
        id
    }

    /// キャッシュキーを生成（操作パターンのハッシュ）
    fn cache_key(&self, output: usize) -> String {
        let mut parts = Vec::new();
        self.build_cache_key(output, &mut parts);
        parts.join("_")
    }

    fn build_cache_key(&self, node_id: usize, parts: &mut Vec<String>) {
        match &self.nodes[node_id].kind {
            NodeKind::Leaf(_) => {
                parts.push(format!("L{}", node_id));
            }
            NodeKind::Op { op, inputs } => {
                for &inp in inputs {
                    self.build_cache_key(inp, parts);
                }
                parts.push(op.cache_key());
            }
        }
    }

    /// CodeGenNode リスト + リーフテンソルを生成（トポロジカル順）
    /// 返値: (リーフテンソル列, CodeGenNodeリスト, 出力ノードID)
    fn to_codegen_nodes(&self, output: usize) -> (Vec<Arc<MetalTensor>>, Vec<CodeGenNode>, usize) {
        // パス1: リーフを全収集（順序確定）
        let mut leaves = Vec::new();
        let mut leaf_id_map = std::collections::HashMap::new();
        self.collect_all_leaves(output, &mut leaves, &mut leaf_id_map);

        let leaf_count = leaves.len();

        // パス2: opノードを生成（リーフ数固定）
        let mut codegen_nodes = Vec::new();
        let mut node_map = std::collections::HashMap::new();
        // リーフのマッピングを node_map に転記
        for (&graph_id, &codegen_id) in &leaf_id_map {
            node_map.insert(graph_id, codegen_id);
        }
        let output_id = self.build_codegen_ops(
            output, leaf_count, &mut codegen_nodes, &mut node_map,
        );
        (leaves, codegen_nodes, output_id)
    }

    /// パス1: リーフを再帰的に収集
    fn collect_all_leaves(
        &self,
        node_id: usize,
        leaves: &mut Vec<Arc<MetalTensor>>,
        leaf_id_map: &mut std::collections::HashMap<usize, usize>,
    ) {
        if leaf_id_map.contains_key(&node_id) {
            return;
        }
        match &self.nodes[node_id].kind {
            NodeKind::Leaf(tensor) => {
                let idx = leaves.len();
                leaves.push(tensor.clone());
                leaf_id_map.insert(node_id, idx);
            }
            NodeKind::Op { inputs, .. } => {
                for &inp in inputs {
                    self.collect_all_leaves(inp, leaves, leaf_id_map);
                }
            }
        }
    }

    /// パス2: op ノードを生成（リーフ数固定前提）
    fn build_codegen_ops(
        &self,
        node_id: usize,
        leaf_count: usize,
        codegen_nodes: &mut Vec<CodeGenNode>,
        node_map: &mut std::collections::HashMap<usize, usize>,
    ) -> usize {
        if let Some(&mapped) = node_map.get(&node_id) {
            return mapped;
        }
        match &self.nodes[node_id].kind {
            NodeKind::Leaf(_) => {
                unreachable!("All leaves should already be in node_map");
            }
            NodeKind::Op { op, inputs } => {
                let mapped_inputs: Vec<usize> = inputs.iter()
                    .map(|&inp| self.build_codegen_ops(inp, leaf_count, codegen_nodes, node_map))
                    .collect();
                let new_id = leaf_count + codegen_nodes.len();
                codegen_nodes.push(CodeGenNode {
                    op: op.clone(),
                    inputs: mapped_inputs,
                });
                node_map.insert(node_id, new_id);
                new_id
            }
        }
    }
}

// ============================================================
// LazyTensor: 遅延実行テンソル
// ============================================================

/// 遅延実行テンソル
///
/// 操作を記録するだけで GPU 実行はしない。
/// `materialize()` で融合カーネルを生成・実行して `MetalTensor` を得る。
pub struct LazyTensor {
    graph: Arc<std::sync::Mutex<OpGraph>>,
    node_id: usize,
}

impl LazyTensor {
    /// 実テンソルから LazyTensor を作成（リーフノード）
    pub fn from_tensor(tensor: MetalTensor) -> Self {
        let mut graph = OpGraph::new();
        let id = graph.add_leaf(Arc::new(tensor));
        LazyTensor {
            graph: Arc::new(std::sync::Mutex::new(graph)),
            node_id: id,
        }
    }

    /// 単項操作を追加
    fn unary_op(&self, op: ElementWiseOp) -> Self {
        let mut graph = self.graph.lock().unwrap();
        let id = graph.add_op(op, vec![self.node_id]);
        LazyTensor {
            graph: self.graph.clone(),
            node_id: id,
        }
    }

    /// 二項操作を追加
    fn binary_op(&self, other: &LazyTensor, op: ElementWiseOp) -> Self {
        let mut graph = self.graph.lock().unwrap();

        // other が別のグラフの場合、other のリーフテンソルを self.graph にマージ
        let other_node_id = if !Arc::ptr_eq(&self.graph, &other.graph) {
            let other_graph = other.graph.lock().unwrap();
            // other のノードから実テンソルを取得
            self.import_node(&mut graph, &other_graph, other.node_id)
        } else {
            other.node_id
        };

        let id = graph.add_op(op, vec![self.node_id, other_node_id]);
        LazyTensor {
            graph: self.graph.clone(),
            node_id: id,
        }
    }

    /// 他グラフのノードを再帰的に self.graph にインポート
    fn import_node(
        &self,
        dest: &mut OpGraph,
        src: &OpGraph,
        src_node_id: usize,
    ) -> usize {
        match &src.nodes[src_node_id].kind {
            NodeKind::Leaf(tensor) => {
                dest.add_leaf(tensor.clone())
            }
            NodeKind::Op { op, inputs } => {
                let new_inputs: Vec<usize> = inputs.iter()
                    .map(|&inp| self.import_node(dest, src, inp))
                    .collect();
                dest.add_op(op.clone(), new_inputs)
            }
        }
    }

    // 単項操作
    pub fn neg(&self) -> Self { self.unary_op(ElementWiseOp::Neg) }
    pub fn abs(&self) -> Self { self.unary_op(ElementWiseOp::Abs) }
    pub fn exp(&self) -> Self { self.unary_op(ElementWiseOp::Exp) }
    pub fn log(&self) -> Self { self.unary_op(ElementWiseOp::Log) }
    pub fn sqrt(&self) -> Self { self.unary_op(ElementWiseOp::Sqrt) }
    pub fn sin(&self) -> Self { self.unary_op(ElementWiseOp::Sin) }
    pub fn cos(&self) -> Self { self.unary_op(ElementWiseOp::Cos) }
    pub fn tanh(&self) -> Self { self.unary_op(ElementWiseOp::Tanh) }
    pub fn sigmoid(&self) -> Self { self.unary_op(ElementWiseOp::Sigmoid) }
    pub fn relu(&self) -> Self { self.unary_op(ElementWiseOp::Relu) }
    pub fn gelu(&self) -> Self { self.unary_op(ElementWiseOp::Gelu) }
    pub fn silu(&self) -> Self { self.unary_op(ElementWiseOp::Silu) }

    // 二項操作
    pub fn add(&self, other: &Self) -> Self { self.binary_op(other, ElementWiseOp::Add) }
    pub fn sub(&self, other: &Self) -> Self { self.binary_op(other, ElementWiseOp::Sub) }
    pub fn mul(&self, other: &Self) -> Self { self.binary_op(other, ElementWiseOp::Mul) }
    pub fn div(&self, other: &Self) -> Self { self.binary_op(other, ElementWiseOp::Div) }
    pub fn pow(&self, other: &Self) -> Self { self.binary_op(other, ElementWiseOp::Pow) }

    // スカラー操作
    pub fn add_scalar(&self, s: f32) -> Self { self.unary_op(ElementWiseOp::AddScalar(s)) }
    pub fn mul_scalar(&self, s: f32) -> Self { self.unary_op(ElementWiseOp::MulScalar(s)) }

    /// 遅延操作を実行し、MetalTensor を返す。
    ///
    /// 1. 操作チェーンのキャッシュキーを生成
    /// 2. キャッシュにあればそのパイプラインを使用
    /// 3. なければ MSL を生成 → コンパイル → キャッシュ
    /// 4. 融合カーネルを実行
    pub fn materialize(&self) -> MetalTensor {
        let graph = self.graph.lock().unwrap();

        // 単一リーフの場合はそのまま返す
        if let NodeKind::Leaf(ref tensor) = graph.nodes[self.node_id].kind {
            return tensor.as_ref().clone();
        }

        let cache_key = graph.cache_key(self.node_id);
        let (leaves, codegen_nodes, output_id) = graph.to_codegen_nodes(self.node_id);
        let leaf_count = leaves.len();
        let shape = graph.nodes[self.node_id].shape.clone();
        let count = shape.iter().product::<usize>();
        drop(graph); // ロックを解放

        // パイプライン取得（キャッシュ or コンパイル）
        let mut cache = get_cache();
        let pipeline_ptr = if let Some(pipeline) = cache.get(&cache_key) {
            pipeline as *const metal::ComputePipelineState
        } else {
            let kernel_name = cache.next_kernel_name();
            let msl_source = MslCodeGen::generate(
                leaf_count, &codegen_nodes, output_id, &kernel_name,
            );


            // ランタイムコンパイル
            let device = get_device();
            let options = metal::CompileOptions::new();
            let library = device.device()
                .new_library_with_source(&msl_source, &options)
                .expect(&format!("Failed to compile fused kernel:\n{}", msl_source));
            let function = library
                .get_function(&kernel_name, None)
                .expect("Failed to get fused kernel function");
            let pipeline = device.device()
                .new_compute_pipeline_state_with_function(&function)
                .expect("Failed to create fused pipeline");

            cache.insert(cache_key.clone(), pipeline);
            cache.get(&cache_key).unwrap() as *const metal::ComputePipelineState
        };
        drop(cache);

        // 出力バッファ作成
        let output = MetalTensor::uninit(&shape, DType::F32);
        let (grid, tpg) = compute_thread_groups(count, unsafe { &*pipeline_ptr });

        // バッファポインタを収集
        let leaf_bufs: Vec<*const metal::Buffer> = leaves.iter()
            .map(|t| t.buffer() as *const metal::Buffer)
            .collect();
        let out_buf = output.buffer() as *const metal::Buffer;
        let count_u32 = count as u32;
        let count_ptr = &count_u32 as *const u32;

        stream_encode(move |encoder| {
            unsafe {
                encoder.set_compute_pipeline_state(&*pipeline_ptr);
                for (i, buf) in leaf_bufs.iter().enumerate() {
                    encoder.set_buffer(i as u64, Some(&**buf), 0);
                }
                encoder.set_buffer(leaf_bufs.len() as u64, Some(&*out_buf), 0);
                encoder.set_bytes(
                    (leaf_bufs.len() + 1) as u64,
                    std::mem::size_of::<u32>() as u64,
                    count_ptr as *const _,
                );
            }
            encoder.dispatch_thread_groups(grid, tpg);
        });

        output
    }
}
