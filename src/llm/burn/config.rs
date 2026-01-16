use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnLLMConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub max_length: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub vocab_size: usize,
}

impl Default for BurnLLMConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("model.onnx"),
            tokenizer_path: PathBuf::from("tokenizer.json"),
            max_length: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            eos_token_id: 151645,  // Qwen3 default
            pad_token_id: 151643,  // Qwen3 default
            vocab_size: 151936,    // Qwen3 default
        }
    }
}

impl BurnLLMConfig {
    pub fn new(model_path: impl Into<PathBuf>, tokenizer_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            tokenizer_path: tokenizer_path.into(),
            ..Default::default()
        }
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = repetition_penalty;
        self
    }

    pub fn with_eos_token_id(mut self, eos_token_id: u32) -> Self {
        self.eos_token_id = eos_token_id;
        self
    }

    pub fn with_pad_token_id(mut self, pad_token_id: u32) -> Self {
        self.pad_token_id = pad_token_id;
        self
    }

    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    pub fn for_qwen3() -> Self {
        Self {
            eos_token_id: 151645,
            pad_token_id: 151643,
            vocab_size: 151936,
            ..Default::default()
        }
    }

    pub fn for_gpt2() -> Self {
        Self {
            eos_token_id: 50256,
            pad_token_id: 50256,
            vocab_size: 50257,
            ..Default::default()
        }
    }
}
