//! Qwen3 model wrapper for burn inference
//! 
//! This module provides a wrapper around the generated burn model
//! that handles KV cache management and autoregressive generation.

use burn::prelude::*;
use burn::tensor::backend::Backend;

/// Qwen3 model configuration
#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        // Qwen3-0.6B configuration
        Self {
            num_layers: 28,
            hidden_size: 1024,
            num_heads: 16,
            head_dim: 64,
            vocab_size: 151936,
        }
    }
}

/// KV Cache for autoregressive generation
pub struct KVCache<B: Backend> {
    pub keys: Vec<Tensor<B, 4>>,
    pub values: Vec<Tensor<B, 4>>,
}

impl<B: Backend> KVCache<B> {
    /// Create empty KV cache for first token
    pub fn empty(config: &Qwen3Config, batch_size: usize, device: &B::Device) -> Self {
        let keys: Vec<_> = (0..config.num_layers)
            .map(|_| {
                Tensor::zeros([batch_size, config.num_heads, 0, config.head_dim], device)
            })
            .collect();
        
        let values: Vec<_> = (0..config.num_layers)
            .map(|_| {
                Tensor::zeros([batch_size, config.num_heads, 0, config.head_dim], device)
            })
            .collect();
        
        Self { keys, values }
    }
    
    /// Get sequence length from cache
    pub fn seq_len(&self) -> usize {
        if self.keys.is_empty() {
            0
        } else {
            self.keys[0].dims()[2]
        }
    }
}

/// Wrapper trait for the generated Qwen3 model
/// This abstracts over the complex forward signature
pub trait Qwen3Forward<B: Backend> {
    /// Single step forward pass
    fn forward_step(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2, Int>,
        position_ids: Tensor<B, 2, Int>,
        kv_cache: KVCache<B>,
    ) -> (Tensor<B, 3>, KVCache<B>);
}

/// Autoregressive generation helper
pub struct Qwen3Generator<B: Backend> {
    config: Qwen3Config,
    device: B::Device,
}

impl<B: Backend> Qwen3Generator<B> {
    pub fn new(config: Qwen3Config, device: B::Device) -> Self {
        Self { config, device }
    }
    
    /// Generate tokens autoregressively
    pub fn generate<M: Qwen3Forward<B>>(
        &self,
        model: &M,
        input_ids: Vec<i64>,
        max_new_tokens: usize,
        eos_token_id: i64,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Vec<i64> {
        let batch_size = 1;
        let mut generated = input_ids.clone();
        let mut kv_cache = KVCache::empty(&self.config, batch_size, &self.device);
        
        // First forward pass with all input tokens
        let input_tensor: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(input_ids.clone(), [1, input_ids.len()]),
            &self.device,
        );
        
        let seq_len = input_ids.len();
        let attention_mask: Tensor<B, 2, Int> = Tensor::ones([1, seq_len], &self.device);
        let position_ids: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new((0..seq_len as i64).collect::<Vec<_>>(), [1, seq_len]),
            &self.device,
        );
        
        let (logits, new_cache) = model.forward_step(
            input_tensor,
            attention_mask,
            position_ids,
            kv_cache,
        );
        kv_cache = new_cache;
        
        // Sample first new token
        let next_token = self.sample_token(&logits, temperature, top_p, top_k);
        generated.push(next_token);
        
        if next_token == eos_token_id {
            return generated;
        }
        
        // Continue autoregressive generation
        for _ in 1..max_new_tokens {
            let cache_len = kv_cache.seq_len();
            
            let input_tensor: Tensor<B, 2, Int> = Tensor::from_data(
                TensorData::new(vec![next_token], [1, 1]),
                &self.device,
            );
            
            let attention_mask: Tensor<B, 2, Int> = Tensor::ones([1, cache_len + 1], &self.device);
            let position_ids: Tensor<B, 2, Int> = Tensor::from_data(
                TensorData::new(vec![cache_len as i64], [1, 1]),
                &self.device,
            );
            
            let (logits, new_cache) = model.forward_step(
                input_tensor,
                attention_mask,
                position_ids,
                kv_cache,
            );
            kv_cache = new_cache;
            
            let next_token = self.sample_token(&logits, temperature, top_p, top_k);
            generated.push(next_token);
            
            if next_token == eos_token_id {
                break;
            }
        }
        
        generated
    }
    
    fn sample_token(
        &self,
        logits: &Tensor<B, 3>,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> i64 {
        // Get last token logits: [batch, seq, vocab] -> [vocab]
        let last_logits = logits.clone().slice([0..1, (logits.dims()[1] - 1)..logits.dims()[1], 0..logits.dims()[2]]);
        let last_logits = last_logits.reshape([logits.dims()[2]]);
        
        // Convert to CPU for sampling
        let logits_data: Vec<f32> = last_logits.to_data().to_vec().unwrap();
        
        if temperature <= 0.0 {
            // Greedy sampling
            logits_data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64)
                .unwrap_or(0)
        } else {
            self.sample_with_temperature(&logits_data, temperature, top_p, top_k)
        }
    }
    
    fn sample_with_temperature(
        &self,
        logits: &[f32],
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> i64 {
        use rand_local as rand;
        
        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();
        
        // Get indices sorted by logit value (descending)
        let mut indexed: Vec<(usize, f32)> = scaled.iter().cloned().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        
        // Apply top-k
        let top_k_candidates: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();
        
        // Compute softmax
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
                return idx as i64;
            }
        }
        
        0
    }
}
