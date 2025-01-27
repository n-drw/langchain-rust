use aws_sdk_bedrockruntime::{error::SdkError, operation::converse::ConverseError};
use thiserror::Error;

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
