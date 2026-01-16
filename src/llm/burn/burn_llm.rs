use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;

use crate::language_models::{GenerateResult, LLMError};
use crate::schemas::{Message, StreamData};

use super::config::BurnLLMConfig;
use super::errors::BurnLLMError;
use super::tokenizer::BurnTokenizer;

use burn::tensor::backend::Backend;
use rand_local as rand;

pub struct BurnLLM<B: Backend> {
    config: BurnLLMConfig,
    tokenizer: Arc<BurnTokenizer>,
    device: B::Device,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> BurnLLM<B> {
    pub fn new(config: BurnLLMConfig, device: B::Device) -> Result<Self, BurnLLMError> {
        let tokenizer = BurnTokenizer::from_file(&config.tokenizer_path)?;
        
        Ok(Self {
            config,
            tokenizer: Arc::new(tokenizer),
            device,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn with_tokenizer(
        config: BurnLLMConfig,
        tokenizer: BurnTokenizer,
        device: B::Device,
    ) -> Self {
        Self {
            config,
            tokenizer: Arc::new(tokenizer),
            device,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn tokenizer(&self) -> &BurnTokenizer {
        &self.tokenizer
    }

    pub fn config(&self) -> &BurnLLMConfig {
        &self.config
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    fn sample_next_token(&self, logits: &[f32]) -> u32 {
        if self.config.temperature <= 0.0 {
            // Greedy sampling
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(self.config.eos_token_id)
        } else {
            // Temperature sampling with top-p/top-k
            self.sample_with_temperature(logits)
        }
    }

    fn sample_with_temperature(&self, logits: &[f32]) -> u32 {
        let temperature = self.config.temperature;
        let top_k = self.config.top_k;
        let top_p = self.config.top_p;

        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();

        // Get indices sorted by logit value (descending)
        let mut indexed: Vec<(usize, f32)> = scaled.iter().cloned().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        // Apply top-k
        let top_k_candidates: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();

        // Compute softmax for top-k candidates
        let max_logit = top_k_candidates
            .iter()
            .map(|(_, l)| *l)
            .fold(f32::NEG_INFINITY, f32::max);
        
        let exp_sum: f32 = top_k_candidates
            .iter()
            .map(|(_, l)| (l - max_logit).exp())
            .sum();
        
        let probs: Vec<(usize, f32)> = top_k_candidates
            .iter()
            .map(|(idx, l)| (*idx, (l - max_logit).exp() / exp_sum))
            .collect();

        // Apply top-p (nucleus sampling)
        let mut cumsum = 0.0;
        let mut filtered: Vec<(usize, f32)> = Vec::new();
        for (idx, prob) in probs {
            cumsum += prob;
            filtered.push((idx, prob));
            if cumsum >= top_p {
                break;
            }
        }

        // Re-normalize
        let total: f32 = filtered.iter().map(|(_, p)| p).sum();
        let normalized: Vec<(usize, f32)> = filtered
            .iter()
            .map(|(idx, p)| (*idx, p / total))
            .collect();

        // Sample
        let r: f32 = rand::random();
        let mut cumsum = 0.0;
        for (idx, prob) in normalized {
            cumsum += prob;
            if r <= cumsum {
                return idx as u32;
            }
        }

        self.config.eos_token_id
    }

    fn apply_repetition_penalty(&self, logits: &mut [f32], generated_ids: &[u32]) {
        let penalty = self.config.repetition_penalty;
        for &id in generated_ids {
            if (id as usize) < logits.len() {
                let logit = &mut logits[id as usize];
                if *logit > 0.0 {
                    *logit /= penalty;
                } else {
                    *logit *= penalty;
                }
            }
        }
    }
}

impl<B: Backend> Clone for BurnLLM<B>
where
    B::Device: Clone,
{
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            tokenizer: Arc::clone(&self.tokenizer),
            device: self.device.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<B: Backend + 'static> crate::language_models::llm::LLM for BurnLLM<B>
where
    B::Device: Clone + Send + Sync,
{
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let input_ids = self
            .tokenizer
            .encode_chat_template(messages)
            .map_err(|e| LLMError::OtherError(e.to_string()))?;

        // TODO: Implement actual ONNX model inference with Burn
        // For now, return a placeholder indicating the model needs to be loaded
        // This will be replaced with actual inference once the ONNX model is integrated
        
        let generation = format!(
            "[BurnLLM] Model inference not yet implemented. \
             Input tokens: {}, Config: max_length={}, temp={}",
            input_ids.len(),
            self.config.max_length,
            self.config.temperature
        );

        Ok(GenerateResult {
            generation,
            tokens: None,
        })
    }

    async fn stream(
        &self,
        _messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        // Streaming will be implemented after basic inference works
        Err(LLMError::OtherError(
            "Streaming not yet implemented for BurnLLM".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = BurnLLMConfig::default();
        assert_eq!(config.max_length, 512);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.vocab_size, 151936); // Qwen3 default
    }

    #[test]
    fn test_config_builder() {
        let config = BurnLLMConfig::default()
            .with_max_length(1024)
            .with_temperature(0.5)
            .with_top_p(0.95);
        
        assert_eq!(config.max_length, 1024);
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.top_p, 0.95);
    }

    #[test]
    fn test_qwen3_preset() {
        let config = BurnLLMConfig::for_qwen3();
        assert_eq!(config.eos_token_id, 151645);
        assert_eq!(config.pad_token_id, 151643);
        assert_eq!(config.vocab_size, 151936);
    }

    #[test]
    fn test_gpt2_preset() {
        let config = BurnLLMConfig::for_gpt2();
        assert_eq!(config.eos_token_id, 50256);
        assert_eq!(config.vocab_size, 50257);
    }
}
