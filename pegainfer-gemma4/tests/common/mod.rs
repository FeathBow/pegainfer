use std::sync::Arc;

use vllm_text::Error;
use vllm_text::Result;
use vllm_text::backend::hf::ResolvedModelFiles;
use vllm_text::backend::hf::TokenizerSource;
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::tokenizer::HuggingFaceTokenizer;
use vllm_text::tokenizer::TekkenTokenizer;
use vllm_text::tokenizer::TiktokenTokenizer;

pub(crate) fn load_tokenizer(model_path: &str) -> DynTokenizer {
    try_load_tokenizer(model_path)
        .unwrap_or_else(|err| panic!("Failed to load tokenizer for {model_path}: {err}"))
}

fn try_load_tokenizer(model_path: &str) -> Result<DynTokenizer> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(Error::Tokenizer(
            "load_tokenizer cannot be called from inside an active Tokio runtime".to_string(),
        ));
    }
    // Local model resolution is exposed as async, so bridge it with a
    // current-thread runtime rather than requiring callers to be async.
    let files = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| {
            Error::Tokenizer(format!("failed to build tokenizer resolver runtime: {err}"))
        })?
        .block_on(ResolvedModelFiles::new(model_path))?;
    match &files.tokenizer {
        TokenizerSource::HuggingFace(path) => Ok(Arc::new(HuggingFaceTokenizer::new(path)?)),
        TokenizerSource::Tiktoken(path) => Ok(Arc::new(TiktokenTokenizer::new(path)?)),
        TokenizerSource::Tekken(path) => Ok(Arc::new(TekkenTokenizer::new(path)?)),
    }
}
