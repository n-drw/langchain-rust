use super::errors::BurnLLMError;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

pub struct BurnTokenizer {
    tokenizer: HfTokenizer,
}

impl BurnTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, BurnLLMError> {
        let tokenizer = HfTokenizer::from_file(path.as_ref())
            .map_err(|e| BurnLLMError::TokenizerError(e.to_string()))?;
        
        Ok(Self { tokenizer })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BurnLLMError> {
        let tokenizer = HfTokenizer::from_bytes(bytes)
            .map_err(|e| BurnLLMError::TokenizerError(e.to_string()))?;
        
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, BurnLLMError> {
        let encoding = self.tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| BurnLLMError::TokenizationError(e.to_string()))?;
        
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, BurnLLMError> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| BurnLLMError::TokenizationError(e.to_string()))
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn encode_chat_template(
        &self,
        messages: &[crate::schemas::Message],
    ) -> Result<Vec<u32>, BurnLLMError> {
        let formatted = self.format_chat_messages(messages);
        self.encode(&formatted, true)
    }

    fn format_chat_messages(&self, messages: &[crate::schemas::Message]) -> String {
        let mut output = String::new();
        
        for msg in messages {
            let role = match msg.message_type {
                crate::schemas::MessageType::SystemMessage => "system",
                crate::schemas::MessageType::AIMessage => "assistant",
                crate::schemas::MessageType::HumanMessage => "user",
                crate::schemas::MessageType::ToolMessage => "tool",
            };
            
            // Qwen3 chat format
            output.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                role, msg.content
            ));
        }
        
        // Add assistant prompt for generation
        output.push_str("<|im_start|>assistant\n");
        
        output
    }
}

impl Clone for BurnTokenizer {
    fn clone(&self) -> Self {
        // HfTokenizer doesn't implement Clone, so we serialize and deserialize
        let json = self.tokenizer.to_string(false).expect("Failed to serialize tokenizer");
        let tokenizer = HfTokenizer::from_bytes(json.as_bytes())
            .expect("Failed to deserialize tokenizer");
        Self { tokenizer }
    }
}
