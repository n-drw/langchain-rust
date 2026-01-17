fn main() {
    #[cfg(feature = "burn-local")]
    burn_onnx_codegen();
}

#[cfg(feature = "burn-local")]
fn burn_onnx_codegen() {
    use burn_import::onnx::ModelGen;
    use std::env;
    use std::path::Path;

    // Check if we have a model path specified via environment variable
    let model_path = env::var("BURN_ONNX_MODEL_PATH").ok();
    
    if let Some(path) = model_path {
        if Path::new(&path).exists() {
            println!("cargo:rerun-if-changed={}", path);
            println!("cargo:rerun-if-env-changed=BURN_ONNX_MODEL_PATH");
            
            ModelGen::new()
                .input(&path)
                .out_dir("src/llm/burn/generated/")
                .run_from_script();
                
            println!("cargo:warning=Generated Burn model from {}", path);
        } else {
            println!("cargo:warning=BURN_ONNX_MODEL_PATH set but file not found: {}", path);
        }
    } else {
        // No model path - skip codegen, use runtime loading instead
        println!("cargo:warning=BURN_ONNX_MODEL_PATH not set, skipping ONNX codegen");
    }
}
