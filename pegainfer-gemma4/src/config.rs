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

#[cfg(all(feature = "gemma4", test))]
pub(crate) fn first_last(layer_types: &[LayerKind], kind: LayerKind) -> Option<(usize, usize)> {
    let mut matching = layer_types
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == kind)
        .map(|(index, _)| index);
    let first = matching.next()?;
    Some((first, matching.next_back().unwrap_or(first)))
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
    // Not manifest inputs: until a serving path lands, only the oracle reads
    // these.
    #[allow(dead_code)]
    pub(crate) rms_norm_eps: f32,
    /// The sliding-attention rope theta; the global family reads its own.
    #[allow(dead_code)]
    pub(crate) sliding_rope_theta: f32,
    #[allow(dead_code)]
    pub(crate) sliding_window: usize,
    #[allow(dead_code)]
    pub(crate) global_rope_theta: f32,
    /// `partial_rotary_factor * global_head_dim`, validated to land on a
    /// positive even width within the head — the active band of the
    /// proportional rope tables.
    #[allow(dead_code)]
    pub(crate) global_rotary_dim: usize,
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
        let rope = tc
            .get("rope_parameters")
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: missing text_config.rope_parameters"))?;
        let sliding_rope = rope
            .get("sliding_attention")
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: missing rope_parameters.sliding_attention"))?;
        let global_rope = rope
            .get("full_attention")
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: missing rope_parameters.full_attention"))?;
        rope_type_field(sliding_rope, "sliding_attention", "default")?;
        rope_type_field(global_rope, "full_attention", "proportional")?;
        let sliding_rope_theta = f32_field(sliding_rope, "sliding_attention", "rope_theta")?;
        let global_rope_theta = f32_field(global_rope, "full_attention", "rope_theta")?;
        anyhow::ensure!(
            sliding_rope_theta > 0.0 && global_rope_theta > 0.0,
            "Gemma 4: rope_theta must be positive (sliding {sliding_rope_theta}, global \
             {global_rope_theta})"
        );
        let global_head_dim = usize_field(tc, "global_head_dim")?;
        let partial = global_rope
            .get("partial_rotary_factor")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Gemma 4: full_attention.partial_rotary_factor missing or not a number"
                )
            })?;
        let rotary = partial * global_head_dim as f64;
        anyhow::ensure!(
            rotary > 0.0
                && rotary.fract() == 0.0
                && rotary as usize <= global_head_dim
                && (rotary as usize).is_multiple_of(2),
            "Gemma 4: partial_rotary_factor {partial} of global_head_dim {global_head_dim} must \
             land on a positive even width within the head, got {rotary}"
        );
        let sliding_window = usize_field(tc, "sliding_window")?;
        anyhow::ensure!(
            sliding_window > 0,
            "Gemma 4: sliding_window must be positive"
        );
        Ok(Self {
            hidden_size: usize_field(tc, "hidden_size")?,
            intermediate_size: usize_field(tc, "intermediate_size")?,
            vocab_size: usize_field(tc, "vocab_size")?,
            num_attention_heads: usize_field(tc, "num_attention_heads")?,
            num_key_value_heads: usize_field(tc, "num_key_value_heads")?,
            num_global_key_value_heads: usize_field(tc, "num_global_key_value_heads")?,
            head_dim: usize_field(tc, "head_dim")?,
            global_head_dim,
            layer_types,
            tie_word_embeddings: bool_field(tc, "tie_word_embeddings")?,
            moe_enabled: bool_field(tc, "enable_moe_block")?,
            rms_norm_eps: f32_field(tc, "text_config", "rms_norm_eps")?,
            sliding_rope_theta,
            sliding_window,
            global_rope_theta,
            global_rotary_dim: rotary as usize,
        })
    }
}

/// Numeric config values land in f32 compute; the checked cast rejects
/// anything the narrowing would turn infinite rather than rounding it in
/// silently.
#[cfg(feature = "gemma4")]
fn f32_field(obj: &serde_json::Value, ctx: &str, field: &str) -> Result<f32> {
    let value = obj
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| anyhow::anyhow!("Gemma 4: {ctx}.{field} missing or not a number"))?;
    let narrowed = value as f32;
    anyhow::ensure!(
        narrowed.is_finite(),
        "Gemma 4: {ctx}.{field} = {value} overflows f32"
    );
    Ok(narrowed)
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

/// `rope_type` selects the table-generation algorithm; a value this engine
/// has not wired for that family must fail here, not silently get the other
/// family's tables.
#[cfg(feature = "gemma4")]
fn rope_type_field(rope_group: &serde_json::Value, ctx: &str, implemented: &str) -> Result<()> {
    let value = rope_group
        .get("rope_type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("Gemma 4: {ctx}.rope_type missing or not a string"))?;
    anyhow::ensure!(
        value == implemented,
        "Gemma 4: {ctx}.rope_type {value:?} is not the implemented {implemented:?}"
    );
    Ok(())
}
