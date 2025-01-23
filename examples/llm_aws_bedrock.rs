// use langchain_rust::embedding::{bedrock::*, embedder_trait::Embedder};

use aws_config::BehaviorVersion;
use langchain_rust::{language_models::llm::LLM, llm::bedrock::Bedrock, schemas::messages};

#[tokio::main]
async fn main() {
    let config = aws_config::defaults(BehaviorVersion::latest()).load().await;
    let bedrock = Bedrock::new(config);
    let t = bedrock.invoke("Test").await;
    println!("{:?}", t);

    // println!("{:#?}", embedder)

    // let azure_config = AzureConfig::default()
    //     .with_api_key("REPLACE_ME_WITH_YOUR_API_KEY")
    //     .with_api_base("https://REPLACE_ME.openai.azure.com")
    //     .with_api_version("2023-05-15")
    //     .with_deployment_id("text-embedding-ada-002");

    // let embedder = OpenAiEmbedder::new(azure_config);
    // let result = embedder.embed_query("Why is the sky blue?").await.unwrap();
    // println!("{:?}", result);
}
