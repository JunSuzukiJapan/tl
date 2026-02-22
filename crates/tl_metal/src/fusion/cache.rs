//! FusionCache — コンパイル済みパイプラインのキャッシュ
//!
//! 操作パターン（キャッシュキー）→ コンパイル済み ComputePipelineState の
//! マッピングを管理。同じ操作チェーンは2回目以降コンパイルコストゼロ。

use metal::ComputePipelineState;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// 融合パイプラインキャッシュ
pub struct FusionCache {
    /// キャッシュキー → コンパイル済みパイプライン
    pipelines: HashMap<String, ComputePipelineState>,
    /// 生成済みカーネル名のカウンタ
    counter: usize,
}

impl FusionCache {
    pub fn new() -> Self {
        FusionCache {
            pipelines: HashMap::new(),
            counter: 0,
        }
    }

    /// キャッシュからパイプラインを取得
    pub fn get(&self, key: &str) -> Option<&ComputePipelineState> {
        self.pipelines.get(key)
    }

    /// パイプラインをキャッシュに挿入
    pub fn insert(&mut self, key: String, pipeline: ComputePipelineState) {
        self.pipelines.insert(key, pipeline);
    }

    /// ユニークなカーネル名を生成
    pub fn next_kernel_name(&mut self) -> String {
        let name = format!("fused_auto_{}", self.counter);
        self.counter += 1;
        name
    }

    /// キャッシュ内のエントリ数
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }
}

/// グローバル FusionCache
static FUSION_CACHE: LazyLock<Mutex<FusionCache>> =
    LazyLock::new(|| Mutex::new(FusionCache::new()));

/// グローバルキャッシュを取得
pub fn get_cache() -> std::sync::MutexGuard<'static, FusionCache> {
    FUSION_CACHE.lock().unwrap()
}
