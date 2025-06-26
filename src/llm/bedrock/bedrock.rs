use async_trait::async_trait;
use aws_config::{BehaviorVersion, SdkConfig};
use aws_sdk_bedrockruntime::{
    operation::invoke_model::InvokeModelOutput,
    operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamOutput,
    Client,
};
use regex::Regex;
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

        let body_bytes = response.body().as_ref();
        let response_str = String::from_utf8_lossy(body_bytes);
        
        let generation = if let Ok(chunk_json) = serde_json::from_str::<serde_json::Value>(&response_str) {
            log::debug!("Bedrock generate response JSON: {}", serde_json::to_string_pretty(&chunk_json).unwrap_or_else(|_| chunk_json.to_string()));
            
            let content = if let Some(output) = chunk_json.get("output").and_then(|o| o.as_str()) {
                output.to_string()
            } else if let Some(text) = chunk_json.get("text").and_then(|t| t.as_str()) {
                text.to_string()
            } else if let Some(generation) = chunk_json.get("generation").and_then(|g| g.as_str()) {
                generation.to_string()
            } else if let Some(choices) = chunk_json.get("choices").and_then(|c| c.as_array()) {
                if let Some(first_choice) = choices.first() {
                    if let Some(text) = first_choice.get("text").and_then(|t| t.as_str()) {
                        text.to_string()
                    } else if let Some(message) = first_choice.get("message") {
                        message.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string()
                    } else {
                        "".to_string()
                    }
                } else {
                    "".to_string()
                }
            } else {
                response_str.to_string()
            };
            
            content
                .replace("<|im_end|>", "")
                .replace("<|im_start|>", "")
                .replace("<|im_number|>", "")
                .replace("<|im_content|>", "")
                .replace("</|im_end|>", "")
                .replace("\n", " ")
                .trim()
                .to_string()
        } else {
            response_str.to_string()
        };
        
        let tokens = if let Ok(chunk_json) = serde_json::from_str::<serde_json::Value>(&response_str) {
            chunk_json.get("usage").and_then(|u| {
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
            })
        } else {
            None
        };
        
        Ok(GenerateResult { tokens, generation })
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let prompt = apply_qwen_chat_template(messages);
    
        let payload = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": true
        });
    
        let body = serde_json::to_string(&payload)
            .map_err(|e| LLMError::OtherError(format!("JSON serialization error: {}", e)))?;
    
        let response = self.client
            .invoke_model_with_response_stream()
            .model_id(self.model_arn.clone())
            .body(body.into_bytes().into())
            .content_type("application/json")
            .accept("application/json")
            .send()
            .await
            .map_err(|e| LLMError::BedrockError(BedrockError::AwsServiceError(Box::new(e))))?;

        fn clean_bedrock_content(content: &str) -> String {
            let tag_re = Regex::new(r"<\|im_[^|>]+?\|>").unwrap();
            tag_re.replace_all(content, "")
                .trim()
                .to_string()
        }
            
        let stream = async_stream::stream! {
            let mut event_stream = response.body;
            while let Ok(Some(event)) = event_stream.recv().await {
                use aws_sdk_bedrockruntime::types::ResponseStream;
                match event {
                    ResponseStream::Chunk(payload_part) => {
                        if let Some(chunk_blob) = payload_part.bytes() {
                            let chunk_bytes = chunk_blob.as_ref();
                            if let Ok(chunk_str) = std::str::from_utf8(chunk_bytes) {
                                if let Ok(chunk_json) = serde_json::from_str::<serde_json::Value>(chunk_str) {
                                    log::debug!("Bedrock stream chunk JSON: {}", serde_json::to_string_pretty(&chunk_json).unwrap_or_else(|_| chunk_json.to_string()));
    
                                    // Extract content as before
                                    let content = if let Some(output) = chunk_json.get("output").and_then(|o| o.as_str()) {
                                        output.to_string()
                                    } else if let Some(text) = chunk_json.get("text").and_then(|t| t.as_str()) {
                                        text.to_string()
                                    } else if let Some(generation) = chunk_json.get("generation").and_then(|g| g.as_str()) {
                                        generation.to_string()
                                    } else if let Some(choices) = chunk_json.get("choices").and_then(|c| c.as_array()) {
                                        if let Some(first_choice) = choices.first() {
                                            if let Some(text) = first_choice.get("text").and_then(|t| t.as_str()) {
                                                text.to_string()
                                            } else if let Some(message) = first_choice.get("message") {
                                                message.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string()
                                            } else {
                                                "".to_string()
                                            }
                                        } else {
                                            "".to_string()
                                        }
                                    } else {
                                        let response_str = chunk_json.as_object()
                                        .and_then(|obj| {
                                            for key in &["response", "message", "answer", "completion", "result", "text"] {
                                                if let Some(value) = obj.get(*key).and_then(|v| v.as_str()) {
                                                    if !value.trim().is_empty() {
                                                        return Some(value.to_string());
                                                    }
                                                }
                                            }
                                            None
                                        })
                                        .unwrap_or_else(|| {
                                            log::warn!("Failed to extract content from Bedrock response. Raw JSON: {}", chunk_str);
                                            "".to_string()
                                        });
                                       clean_bedrock_content(&response_str)
                                    };


                                    let clean_content =  clean_bedrock_content(&content);
    
                                    // Only yield if we have meaningful content
                                    if !clean_content.is_empty() {
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
                                            value: serde_json::Value::String(clean_content.clone()),
                                            tokens: usage,
                                            content: clean_content,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
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
