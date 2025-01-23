use std::sync::Arc;

use crate::embedding::{embedder_trait::Embedder, EmbedderError};
use async_trait::async_trait;

const DEFAULT_MODEL: &str = "nomic-embed-text";

#[derive(Debug)]
pub struct AWSEmbedder {
    pub(crate) client: Arc<_>,
    pub(crate) model: String,
}

impl AWSEmbedder {
    pub fn new<S: Into<String>>(client: Arc<_>, model: S, options: Option<_>) -> Self {
        Self {}
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_options(mut self, options: _) -> Self {
        self.options = Some(options);
        self
    }
}

impl Default for AWSEmbedder {
    fn default() -> Self {
        let client = Arc::new(AWSEmbedder::default());
        Self::new(client, String::from(DEFAULT_MODEL), None)
    }
}

#[async_trait]
impl Embedder for AWSEmbedder {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        log::debug!("Embedding documents: {:?}", documents);

        Ok(embeddings)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        log::debug!("Embedding query: {:?}", text);

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_ollama_embed() {
        let ollama = AWSEmbedder::default()
            .with_model("nomic-embed-text")
            .with_options(GenerationOptions::default().temperature(0.5));

        let response = ollama.embed_query("Why is the sky blue?").await.unwrap();

        assert_eq!(response.len(), 768);
    }
}
