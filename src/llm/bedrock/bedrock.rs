use async_trait::async_trait;
use aws_config::{BehaviorVersion, SdkConfig};
use aws_sdk_bedrockruntime::{
    error::SdkError,
    operation::converse::{ConverseError, ConverseOutput},
    types::{ContentBlock, ConversationRole, Message as BedrockMessage, SystemContentBlock},
    Client,
};
use futures::Stream;
use std::pin::Pin;
use thiserror::Error;

use crate::{
    language_models::{llm::LLM, GenerateResult, LLMError, TokenUsage},
    schemas::{Message, MessageType, StreamData},
};

const DEFAULT_MODEL: &str = "meta.llama3-8b-instruct-v1:0";

// Examples
// https://github.com/awslabs/aws-sdk-rust/tree/main/examples/examples/bedrock-runtime/src/bin

#[derive(Error, Debug)]
pub enum BedrockError {
    #[error("Failed to parse messages: {0}")]
    FailedToParseMessages(String),

    #[error("Failed to parse response: {0}")]
    FailedToParseResponse(String),

    #[error("Failed extract text from response: {0}")]
    FailedToExtractText(String),

    #[error("{0}")]
    AwsServiceError(SdkError<ConverseError>),

    #[error("System message should be sent in separate call")]
    SystemMessageError,

    #[error("Failed to build messages: '{0}'")]
    FailedToBuildMessages(String),
}

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
fn create_bedrock_message(
    message: &Message,
    role: ConversationRole,
) -> Result<BedrockMessage, LLMError> {
    BedrockMessage::builder()
        .role(role)
        .content(ContentBlock::Text(message.content.clone()))
        .build()
        .map_err(|build_error| {
            LLMError::BedrockError(BedrockError::FailedToBuildMessages(build_error.to_string()))
        })
}

#[async_trait]
impl LLM for Bedrock {
    /*
       Questions:
           1. What is the ToolMessage map to?

    */

    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let conversation_builder = self.client.converse().model_id(self.model.clone());

        // Ok(message);
        let system_messages = messages
            .iter()
            .filter(|m| m.message_type == MessageType::SystemMessage)
            .map(|m| SystemContentBlock::Text(m.content.clone()))
            .collect::<Vec<SystemContentBlock>>();

        let ai_messages = messages
            .iter()
            .filter(|m| m.message_type == MessageType::AIMessage)
            // TODO: Not sure if this is correct. And nto sure if it
            // the "AIMessage" is needed for Bedrock
            .map(|m| create_bedrock_message(m, ConversationRole::Assistant))
            .collect::<Result<Vec<BedrockMessage>, LLMError>>()?;

        let human_messages = messages
            .iter()
            .filter(|m| m.message_type == MessageType::HumanMessage)
            .map(|m| create_bedrock_message(m, ConversationRole::User))
            .collect::<Result<Vec<BedrockMessage>, LLMError>>()?;

        let tool_messages = messages
            .iter()
            .filter(|m| m.message_type == MessageType::ToolMessage)
            .map(|m| create_bedrock_message(m, ConversationRole::Assistant))
            .collect::<Result<Vec<BedrockMessage>, LLMError>>()?;

        let response = conversation_builder
            .set_system(Some(system_messages))
            .set_messages(Some(human_messages))
            .send()
            .await
            .map_err(|e| LLMError::BedrockError(BedrockError::AwsServiceError(e)))?;

        let tokens = response
            .clone()
            .usage
            .map(|usage| TokenUsage::new(usage.input_tokens as u32, usage.output_tokens as u32));

        let generation = get_converse_output_text(response).or(Err(LLMError::BedrockError(
            BedrockError::FailedToExtractText("Failed to extract text from response".to_string()),
        )))?;

        Ok(GenerateResult { tokens, generation })
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        todo!()
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
