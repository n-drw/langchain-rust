/// Qwen-style chat template formatting for Bedrock prompt input
/// See: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct and Bedrock docs
///
/// Example output:
/// <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n
use crate::schemas::{Message, MessageType};

pub fn apply_qwen_chat_template(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        let role = match msg.message_type {
            MessageType::SystemMessage => "system",
            MessageType::AIMessage => "assistant",
            MessageType::HumanMessage => "user",
            MessageType::ToolMessage => "user", // fallback
        };
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&msg.content);
        out.push_str("<|im_end|>\n");
    }
    // Add generation prompt for the assistant
    out.push_str("<|im_start|>assistant\n\n");
    out
}
