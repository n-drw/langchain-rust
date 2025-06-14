use async_trait::async_trait;
use aws_config::{BehaviorVersion, SdkConfig};
use aws_sdk_bedrockruntime::{
    operation::invoke_model::InvokeModelOutput,
    Client,
};
use crate::llm::bedrock::qwen_chat_template::apply_qwen_chat_template;
use serde_json;
use futures::Stream;
use std::pin::Pin;

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
    pub fn new(config: SdkConfig, model_arn: String) -> Self {
        let client = Client::new(&config);

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
            Self::new(config, "arn:aws:bedrock:us-west-2:211125612083:imported-model/6acxq2e0nctj".to_string())
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
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate() {
        let config: SdkConfig = aws_config::from_env().region("us-west-2").load().await;
        let bedrock = Bedrock::new(config, "arn:aws:bedrock:us-west-2:211125612083:imported-model/6acxq2e0nctj".to_string());
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
        let bedrock = Bedrock::new(config, "arn:aws:bedrock:us-west-2:211125612083:imported-model/6acxq2e0nctj".to_string());

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
