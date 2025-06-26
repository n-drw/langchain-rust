/// Qwen-style chat template formatting for Bedrock prompt input
/// See: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct and Bedrock docs
///
/// Example output:
///  system
/// You are a helpful assistant.
/// 
/// user
/// Hello!
/// 
/// assistant
/// 
use crate::schemas::{Message, MessageType};

pub fn apply_qwen_chat_template(messages: &[Message], append_static_prompt: bool) -> String {
    let mut out = String::new();
    for msg in messages {
        let role = match msg.message_type {
            MessageType::SystemMessage => "system",
            MessageType::AIMessage => "assistant",
            MessageType::HumanMessage => "user",
            MessageType::ToolMessage => "user", // fallback
        };
        out.push_str(" ");
        out.push_str(role);
        out.push('\n');
        out.push_str(&msg.content);
        out.push_str(" ");
    }
    // Conditionally add generation prompt for the assistant
    if append_static_prompt {
        out.push_str(" system
You are a helpful assistant. 
 user
Hello! 
 assistant
");
    }
    out
}
