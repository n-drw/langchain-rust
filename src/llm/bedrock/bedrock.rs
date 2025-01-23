use aws_config::{BehaviorVersion, SdkConfig};
use aws_sdk_bedrockruntime::{
    operation::converse::{ConverseError, ConverseOutput},
    types::{ContentBlock, ConversationRole, Message as BedrockMessage},
    Client,
};
use futures::Stream;
use log::info;
use std::{pin::Pin, sync::Arc};

use crate::{
    embedding::{embedder_trait::Embedder, EmbedderError},
    language_models::{llm::LLM, GenerateResult, LLMError, TokenUsage},
    schemas::{Message, StreamData},
};
use async_trait::async_trait;

const DEFAULT_MODEL: &str = "meta.llama3-8b-instruct-v1:0";

// Examples
// https://github.com/awslabs/aws-sdk-rust/tree/main/examples/examples/bedrock-runtime/src/bin

#[derive(Debug, Clone)]
pub struct Bedrock {
    pub(crate) client: Client,
    pub(crate) config: SdkConfig,
    pub(crate) model: String,
}

impl Bedrock {
    pub fn new(config: SdkConfig) -> Self {
        let client = Client::new(&config);

        Self {
            client,
            config,
            model: String::from(DEFAULT_MODEL),
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_options(mut self, config: SdkConfig) -> Self {
        self.config = config;
        self
    }
}

impl Default for Bedrock {
    fn default() -> Self {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let config: SdkConfig = aws_config::defaults(BehaviorVersion::latest()).load().await;
            Self::new(config)
        })
    }
}

#[derive(Debug)]
struct BedrockConverseError(String);
impl std::fmt::Display for BedrockConverseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Can't invoke '{}'. Reason: {}", DEFAULT_MODEL, self.0)
    }
}
impl std::error::Error for BedrockConverseError {}
impl From<&str> for BedrockConverseError {
    fn from(value: &str) -> Self {
        BedrockConverseError(value.to_string())
    }
}
impl From<&ConverseError> for BedrockConverseError {
    fn from(value: &ConverseError) -> Self {
        BedrockConverseError::from(match value {
            ConverseError::ModelTimeoutException(_) => "Model took too long",
            ConverseError::ModelNotReadyException(_) => "Model is not ready",
            _ => "Unknown",
        })
    }
}

fn get_converse_output_text(output: ConverseOutput) -> Result<String, BedrockConverseError> {
    let text = output
        .output()
        .ok_or("no output")?
        .as_message()
        .map_err(|_| "output not a message")?
        .content()
        .first()
        .ok_or("no content in message")?
        .as_text()
        .map_err(|_| "content is not text")?
        .to_string();
    Ok(text)
}

#[async_trait]
impl LLM for Bedrock {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let messages = messages
            .iter()
            .map(|message| {
                BedrockMessage::builder()
                    .role(ConversationRole::User)
                    .content(ContentBlock::Text(message.content.clone()))
                    .build()
                    .map_err(|_| "failed to build message")
            })
            .collect::<Result<Vec<BedrockMessage>, &str>>()
            .unwrap(); // TODO: Remove unwrap

        println!("Model: {}", self.model);

        let response = self
            .client
            .converse()
            .model_id(self.model.clone())
            .set_messages(Some(messages))
            .send()
            .await;

        match response {
            Ok(output) => {
                // TODO: Handle 'service error' (it will pop when invalid model is selected)
                let text = get_converse_output_text(output).unwrap(); // TODO: Remove unwrap
                println!("{}", text);
            }
            Err(e) => println!("{}", e),
        }

        let tokens = TokenUsage::default();
        let generation = String::from("test test test generation");

        Ok(GenerateResult {
            tokens: Some(tokens),
            generation,
        })
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        todo!()
    }
    async fn invoke(&self, prompt: &str) -> Result<String, LLMError> {
        self.generate(&[Message::new_human_message(prompt)])
            .await
            .map(|res| res.generation)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[tokio::test]
//     #[ignore]
//     async fn test_ollama_embed() {
//         let ollama = AWSEmbedder::default()
//             .with_model("nomic-embed-text")
//             .with_options(GenerationOptions::default().temperature(0.5));

//         let response = ollama.embed_query("Why is the sky blue?").await.unwrap();

//         assert_eq!(response.len(), 768);
//     }
// }
