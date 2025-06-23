use async_trait::async_trait;
use aws_config::{BehaviorVersion, SdkConfig};
use aws_sdk_bedrockruntime::{
    operation::invoke_model::InvokeModelOutput,
    operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamOutput,
    Client,
};
use crate::{language_models::TokenUsage, llm::bedrock::qwen_chat_template::apply_qwen_chat_template, schemas::convert::OpenAiIntoLangchain};
use serde_json;
use futures::{FutureExt, Stream, StreamExt, TryStreamExt};
use std::pin::Pin;
use async_stream;
use log;

use crate::{
    language_models::{llm::LLM, GenerateResult, LLMError},
    schemas::{Message, StreamData},
};

use super::BedrockError;

const DEFAULT_MODEL: &str = "meta.llama3-8b-instruct-v1:0";

// Examples
// https://github.com/awslabs/aws-sdk-rust/tree/main/examples/examples/bedrock-runtime/src/bin

#[derive(Debug, Clone)]
pub struct Bedrock {
    pub(crate) client: Client,
    pub(crate) config: SdkConfig,
    pub(crate) model_arn: String,
}

impl Bedrock {
    pub fn new(client: Client, config: SdkConfig, model_arn: String) -> Self {
        Self {
            client,
            config,
            model_arn,
        }
    }
}

impl Default for Bedrock {
    fn default() -> Self {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let config: SdkConfig = aws_config::defaults(BehaviorVersion::latest()).load().await;
            let client = Client::new(&config);
            Self::new(client, config, "arn:aws:bedrock:us-west-2:211125612083:imported-model/6acxq2e0nctj".to_string())
        })
    }
}


// Removed: get_converse_output_text and ConverseOutput logic (no longer needed for invoke_model).

// No longer needed: BedrockMessage logic removed for invoke_model flow.

#[async_trait]
impl LLM for Bedrock {
    /*
       Questions:
           1. What is the ToolMessage map to?

    */

    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        // Format messages using Qwen chat template
        use serde_json::json;
        let prompt = apply_qwen_chat_template(messages);
        let body = json!({ "prompt": prompt }).to_string();
        let response: InvokeModelOutput = self.client
            .invoke_model()
            .model_id(self.model_arn.clone())
            .body(body.into_bytes().into())
            .content_type("application/json")
            .accept("application/json")
            .send()
            .await
            .map_err(|e| LLMError::BedrockError(BedrockError::AwsServiceError(Box::new(e))))?;

        // Extract the output from the response body
        let body_bytes = response.body().as_ref();
        let generation = String::from_utf8_lossy(body_bytes).to_string();
        let tokens = None;
        Ok(GenerateResult { tokens, generation })
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        // Format prompt using Qwen chat template
        let prompt = apply_qwen_chat_template(messages);
        
        // Create the request payload
        let payload = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": true
        });
        
        let body = serde_json::to_string(&payload)
            .map_err(|e| LLMError::OtherError(format!("JSON serialization error: {}", e)))?;

        // Send the streaming request to Bedrock
        let response = self.client
            .invoke_model_with_response_stream()
            .model_id(self.model_arn.clone())
            .body(body.into_bytes().into())
            .content_type("application/json")
            .accept("application/json")
            .send()
            .await
            .map_err(|e| LLMError::BedrockError(BedrockError::AwsServiceError(Box::new(e))))?;

        // Process the event stream using the AWS SDK EventReceiver
        let stream = async_stream::stream! {
            let mut event_stream = response.body;
            while let Ok(Some(event)) = event_stream.recv().await {
                // Handle different event types from the response stream
                use aws_sdk_bedrockruntime::types::ResponseStream;
                match event {
                    ResponseStream::Chunk(payload_part) => {
                        // Extract bytes from the chunk
                        if let Some(chunk_blob) = payload_part.bytes() {
                            let chunk_bytes = chunk_blob.as_ref();
                            if let Ok(chunk_str) = std::str::from_utf8(chunk_bytes) {
                                // Parse the JSON response
                                if let Ok(chunk_json) = serde_json::from_str::<serde_json::Value>(chunk_str) {
                                    // Add debug logging to see the actual response structure
                                    log::debug!("Bedrock stream chunk JSON: {}", serde_json::to_string_pretty(&chunk_json).unwrap_or_else(|_| chunk_json.to_string()));
                                    
                                    // Extract content from the response with improved logic for DeepSeek model
                                    let content = if let Some(text) = chunk_json.as_str() {
                                        text.to_string()
                                    } else if let Some(output) = chunk_json.get("output").and_then(|o| o.as_str()) {
                                        output.to_string()
                                    } else if let Some(text) = chunk_json.get("text").and_then(|t| t.as_str()) {
                                        text.to_string()
                                    } else if let Some(generation) = chunk_json.get("generation").and_then(|g| g.as_str()) {
                                        generation.to_string()
                                    } else if let Some(choices) = chunk_json.get("choices").and_then(|c| c.as_array()) {
                                        // Handle OpenAI-style response format (common for custom models)
                                        if let Some(first_choice) = choices.first() {
                                            // First check for direct text field in the choice
                                            if let Some(text) = first_choice.get("text").and_then(|t| t.as_str()) {
                                                text.to_string()
                                            } 
                                            // Then check for delta object
                                            else if let Some(delta) = first_choice.get("delta") {
                                                if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                                    content.to_string()
                                                } else if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                                    text.to_string()
                                                } else {
                                                    "".to_string()
                                                }
                                            } else {
                                                "".to_string()
                                            }
                                        } else {
                                            "".to_string()
                                        }
                                    } else {
                                        // Fallback: check if this is a delta response
                                        if let Some(delta) = chunk_json.get("delta") {
                                            if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                                text.to_string()
                                            } else if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                                content.to_string()
                                            } else {
                                                "".to_string()
                                            }
                                        } else {
                                            // Final fallback: try to find any string field that might contain the content
                                            chunk_json.as_object()
                                                .and_then(|obj| {
                                                    // Look for common content fields
                                                    for key in &["response", "message", "answer", "completion", "result"] {
                                                        if let Some(value) = obj.get(*key).and_then(|v| v.as_str()) {
                                                            if !value.trim().is_empty() {
                                                                return Some(value.to_string());
                                                            }
                                                        }
                                                    }
                                                    None
                                                })
                                                .unwrap_or_else(|| {
                                                    // If we still can't find content, log the structure and return the raw JSON
                                                    log::warn!("Failed to extract content from Bedrock response. Raw JSON: {}", chunk_str);
                                                    // Return a small indicator that we received something
                                                    if chunk_str.len() > 10 {
                                                        format!("[Unparsed response: {}...]", &chunk_str[..50.min(chunk_str.len())])
                                                    } else {
                                                        "".to_string()
                                                    }
                                                })
                                        }
                                    };

                                    // Always yield something, even if content is empty (for debugging)
                                    let should_yield = !content.trim().is_empty() || chunk_json.as_object().map_or(false, |obj| !obj.is_empty());
                                    
                                    if should_yield {
                                        // Extract usage info if available
                                        let usage = chunk_json.get("usage").and_then(|u| {
                                            Some(TokenUsage {
                                                prompt_tokens: u.get("input_tokens")
                                                    .or_else(|| u.get("prompt_tokens"))
                                                    .and_then(|t| t.as_u64())
                                                    .unwrap_or(0) as u32,
                                                completion_tokens: u.get("output_tokens")
                                                    .or_else(|| u.get("completion_tokens"))
                                                    .and_then(|t| t.as_u64())
                                                    .unwrap_or(0) as u32,
                                                total_tokens: u.get("total_tokens")
                                                    .and_then(|t| t.as_u64())
                                                    .unwrap_or(0) as u32,
                                            })
                                        });

                                        yield Ok(StreamData {
                                            value: chunk_json.clone(),
                                            tokens: usage,
                                            content,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        // Handle other event types (metadata, etc.) - typically ignore
                    }
                }
            }
        };
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate() {
        let config: SdkConfig = aws_config::from_env().region("us-west-2").load().await;
        let client = Client::new(&config);
        let bedrock = Bedrock::new(client, config, "arn:aws:bedrock:us-west-2:211125612083:imported-model/6acxq2e0nctj".to_string());
        let messages = vec![
            Message::new_system_message("You are a helpful coding assistant."),
            Message::new_human_message("Say hello to the world!"),
        ];
        let response = bedrock.generate(&messages).await.unwrap();
        println!("{}", response.generation);
        assert!(response.generation.len() > 0);
    }

    #[tokio::test]
    async fn test_generate_with_messages() {
        let config: SdkConfig = aws_config::from_env().region("us-west-2").load().await;
        let client = Client::new(&config);
        let bedrock = Bedrock::new(client, config, "arn:aws:bedrock:us-west-2:211125612083:imported-model/6acxq2e0nctj".to_string());

        let messages = vec![
            Message::new_system_message(
                "You are the voice interface of an overpriced cloud tool, AWS.",
            ),
            Message::new_human_message("What's the point of cloud services?"),
        ];

        let response = bedrock.generate(&messages).await.unwrap();
        println!("{:#?}", response);

        assert!(response.generation.len() > 0);
    }

    // #[tokio::test]
    // #[ignore]
    // async fn test_stream() {
    //     let ollama = Ollama::default().with_model("llama3.2");

    //     let message = Message::new_human_message("Why does water boil at 100 degrees?");
    //     let mut stream = ollama.stream(&vec![message]).await.unwrap();
    //     let mut stdout = tokio::io::stdout();
    //     while let Some(res) = stream.next().await {
    //         let data = res.unwrap();
    //         stdout.write(data.content.as_bytes()).await.unwrap();
    //     }
    //     stdout.write(b"\n").await.unwrap();
    //     stdout.flush().await.unwrap();
    // }
}
