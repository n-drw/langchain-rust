use std::fmt::Debug;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BedrockError {
    #[error("Failed to parse messages: {0}")]
    FailedToParseMessages(String),

    #[error("Failed to parse response: {0}")]
    FailedToParseResponse(String),

    #[error("Failed extract text from response: {0}")]
    FailedToExtractText(String),

    #[error("AWS service error: {0:?}")]
    AwsServiceError(Box<dyn std::error::Error + Send + Sync + 'static>),

    #[error("System message should be sent in separate call")]
    SystemMessageError,

    #[error("Failed to build messages: '{0}'")]
    FailedToBuildMessages(String),
}
