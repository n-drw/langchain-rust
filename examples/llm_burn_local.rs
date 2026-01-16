use langchain_rust::language_models::llm::LLM;
use langchain_rust::llm::burn::{BurnLLM, BurnLLMConfig, BurnTokenizer};
use langchain_rust::schemas::Message;

use burn_ndarray::NdArray;

type Backend = NdArray<f32>;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path to the exported Qwen3-0.6B model
    // Update these paths to point to your exported model
    let model_path = std::env::var("BURN_MODEL_PATH")
        .unwrap_or_else(|_| "../hello_michi/onnx_models/qwen3-0.6b/model.onnx".to_string());
    let tokenizer_path = std::env::var("BURN_TOKENIZER_PATH")
        .unwrap_or_else(|_| "../hello_michi/onnx_models/qwen3-0.6b/tokenizer.json".to_string());

    println!("=== Burn Local LLM Example ===\n");
    println!("Model path: {}", model_path);
    println!("Tokenizer path: {}", tokenizer_path);

    // Load tokenizer first to test it works
    println!("\n1. Loading tokenizer...");
    let tokenizer = match BurnTokenizer::from_file(&tokenizer_path) {
        Ok(t) => {
            println!("   ✅ Tokenizer loaded successfully");
            println!("   Vocab size: {}", t.vocab_size());
            t
        }
        Err(e) => {
            eprintln!("   ❌ Failed to load tokenizer: {}", e);
            return Err(e.into());
        }
    };

    // Test tokenization
    println!("\n2. Testing tokenization...");
    let test_text = "Hello, how are you?";
    match tokenizer.encode(test_text, true) {
        Ok(ids) => {
            println!("   Input: \"{}\"", test_text);
            println!("   Token IDs: {:?}", ids);
            println!("   Token count: {}", ids.len());

            // Decode back
            if let Ok(decoded) = tokenizer.decode(&ids, true) {
                println!("   Decoded: \"{}\"", decoded);
            }
        }
        Err(e) => {
            eprintln!("   ❌ Tokenization failed: {}", e);
        }
    }

    println!("\n3. Testing chat template...");
    let messages = vec![
        Message::new_system_message("You are Michi, a helpful AI assistant."),
        Message::new_human_message("What is the capital of France?"),
    ];

    match tokenizer.encode_chat_template(&messages) {
        Ok(ids) => {
            println!("   Chat template token count: {}", ids.len());
            println!("   First 20 tokens: {:?}", &ids[..ids.len().min(20)]);
        }
        Err(e) => {
            eprintln!("   ❌ Chat template encoding failed: {}", e);
        }
    }

    println!("\n4. Creating Burn configuration...");
    let config = BurnLLMConfig::for_qwen3()
        .with_max_length(256)
        .with_temperature(0.7)
        .with_top_p(0.9)
        .with_top_k(50);

    println!("   Max length: {}", config.max_length);
    println!("   Temperature: {}", config.temperature);
    println!("   Top-p: {}", config.top_p);
    println!("   Top-k: {}", config.top_k);
    println!("   EOS token ID: {}", config.eos_token_id);
    println!("   Vocab size: {}", config.vocab_size);

    println!("\n5. Creating Burn instance...");
    let device = Default::default();
    let llm: BurnLLM<Backend> = BurnLLM::with_tokenizer(
        BurnLLMConfig::new(&model_path, &tokenizer_path)
            .with_max_length(256)
            .with_temperature(0.7),
        tokenizer,
        device,
    );
    println!("   ✅ BurnLLM created");

    println!("\n6. Testing LLM trait (generate)...");
    let messages = vec![
        Message::new_system_message("You are Michi, a helpful AI assistant."),
        Message::new_human_message("Say hello!"),
    ];

    match llm.generate(&messages).await {
        Ok(result) => {
            println!("   Response: {}", result.generation);
        }
        Err(e) => {
            eprintln!("   ❌ Generation failed: {}", e);
        }
    }

    println!("\n=== Example Complete ===");
    println!("\nNote: Full ONNX model inference is not yet implemented.");
    println!("The tokenizer and configuration are working correctly.");
    println!("Next steps:");
    println!("  1. Integrate burn-import for ONNX model loading");
    println!("  2. Implement forward pass with KV caching");
    println!("  3. Add autoregressive generation loop");

    Ok(())
}
