use thiserror::Error;

#[derive(Error, Debug)]
pub enum BurnLLMError {
    #[error("Failed to load ONNX model: {0}")]
    ModelLoadError(String),

    #[error("Failed to load tokenizer: {0}")]
    TokenizerError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Backend initialization error: {0}")]
    BackendError(String),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("Model not initialized")]
    ModelNotInitialized,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl From<BurnLLMError> for crate::language_models::LLMError {
    fn from(err: BurnLLMError) -> Self {
        crate::language_models::LLMError::OtherError(err.to_string())
    }
}
