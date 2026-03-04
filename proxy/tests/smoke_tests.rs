use config::Config;
use inference::ModelEnvironment;

#[test]
fn test_config_loads_defaults() {
    let config = Config::default();
    assert!(config.disabled_entities.is_empty());
}

#[test]
fn test_model_env_missing_fails() {
    // Ensuring fail-closed behavior on missing env vars
    // Before running, let's unset just in case they are leaking from somewhere
    unsafe {
        std::env::remove_var("NER_MODEL_PATH");
        std::env::remove_var("NER_TOKENIZER_PATH");
    }
    
    let res = ModelEnvironment::load();
    assert!(res.is_err(), "ModelEnvironment should fail to load without env vars");
}
