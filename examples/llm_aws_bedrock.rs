// use langchain_rust::embedding::{bedrock::*, embedder_trait::Embedder};

use aws_config::BehaviorVersion;
use langchain_rust::{
    language_models::llm::LLM,
    llm::bedrock::Bedrock,
    schemas::{messages, Message},
};

#[tokio::main]
async fn main() {
    let config = aws_config::defaults(BehaviorVersion::latest()).load().await;
    let system_message = Message::new_system_message("You are a beautiful unicorn.");
    let user_message = Message::new_human_message("Hello, how are you?");
    let messages = vec![system_message, user_message];
    let bedrock = Bedrock::new(config);
    let t = bedrock.generate(&messages).await;
    // let t = bedrock.invoke("test").await;

    println!("{:#?}", t);

    // let azure_config = AzureConfig::default()
    //     .with_api_key("REPLACE_ME_WITH_YOUR_API_KEY")
    //     .with_api_base("https://REPLACE_ME.openai.azure.com")
    //     .with_api_version("2023-05-15")
    //     .with_deployment_id("text-embedding-ada-002");

    // let embedder = OpenAiEmbedder::new(azure_config);
    // let result = embedder.embed_query("Why is the sky blue?").await.unwrap();
    // println!("{:?}", result);
}
