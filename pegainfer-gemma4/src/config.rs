//! Typed text-tower config, read only from a file the probe has accepted.

#[cfg(feature = "gemma4")]
use anyhow::Result;
#[cfg(feature = "gemma4")]
use anyhow::bail;

// Only the loader reads a config off disk.
#[cfg(feature = "gemma4")]
use crate::probe::probe_config_json;

/// Selects the head dim, the KV head count, and whether the layer has a
/// `v_proj`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LayerKind {
    Sliding,
    Global,
}

/// What the manifest is derived from. Only [`Gemma4Config::from_file`] is
/// probe-backed; a value built directly is not, so consumers check what they
/// depend on.
#[derive(Clone, Debug)]
pub(crate) struct Gemma4Config {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) vocab_size: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    /// 1, 2 or 4 by size, and not derivable from `num_key_value_heads`.
    pub(crate) num_global_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) global_head_dim: usize,
    pub(crate) layer_types: Vec<LayerKind>,
    pub(crate) tie_word_embeddings: bool,
    /// The MoE size keeps its dense MLP and adds experts alongside it.
    pub(crate) moe_enabled: bool,
}

#[cfg(feature = "gemma4")]
impl Gemma4Config {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let path = format!("{model_path}/config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("Gemma 4: cannot read {path}: {e}"))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("Gemma 4: {path} is not valid JSON: {e}"))?;
        probe_config_json(&json)?;
        Self::from_json(&json)
    }

    fn from_json(json: &serde_json::Value) -> Result<Self> {
        let tc = json
            .get("text_config")
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: missing text_config"))?;
        let layer_types = tc
            .get("layer_types")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: missing text_config.layer_types"))?
            .iter()
            .map(|entry| match entry.as_str() {
                Some("sliding_attention") => Ok(LayerKind::Sliding),
                Some("full_attention") => Ok(LayerKind::Global),
                other => bail!("Gemma 4: unexpected layer type {other:?}"),
            })
            .collect::<Result<Vec<_>>>()?;
        let num_hidden_layers = usize_field(tc, "num_hidden_layers")?;
        anyhow::ensure!(
            layer_types.len() == num_hidden_layers,
            "Gemma 4: layer_types has {} entries but num_hidden_layers is {num_hidden_layers}",
            layer_types.len()
        );
        Ok(Self {
            hidden_size: usize_field(tc, "hidden_size")?,
            intermediate_size: usize_field(tc, "intermediate_size")?,
            vocab_size: usize_field(tc, "vocab_size")?,
            num_attention_heads: usize_field(tc, "num_attention_heads")?,
            num_key_value_heads: usize_field(tc, "num_key_value_heads")?,
            num_global_key_value_heads: usize_field(tc, "num_global_key_value_heads")?,
            head_dim: usize_field(tc, "head_dim")?,
            global_head_dim: usize_field(tc, "global_head_dim")?,
            layer_types,
            tie_word_embeddings: bool_field(tc, "tie_word_embeddings")?,
            moe_enabled: bool_field(tc, "enable_moe_block")?,
        })
    }
}

#[cfg(feature = "gemma4")]
fn usize_field(text_config: &serde_json::Value, field: &str) -> Result<usize> {
    let value = text_config
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            anyhow::anyhow!("Gemma 4: text_config.{field} missing or not a non-negative integer")
        })?;
    usize::try_from(value)
        .map_err(|_| anyhow::anyhow!("Gemma 4: text_config.{field} = {value} does not fit usize"))
}

#[cfg(feature = "gemma4")]
fn bool_field(text_config: &serde_json::Value, field: &str) -> Result<bool> {
    text_config
        .get(field)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| anyhow::anyhow!("Gemma 4: text_config.{field} missing or not a boolean"))
}
