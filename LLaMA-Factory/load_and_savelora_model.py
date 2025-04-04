import sys
sys.path.append("src")
from src import load_pretrained, ModelArguments

if __name__ == "__main__":
    path_to_save_model = './lora/Qwen2.5-7B-Instruct-lora-dpo-lr-1e-5-beta0.1/'
    path_to_lora_checkpoint = '/map-vepfs/siwei/coig/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/lora/dpo_error'
    model_args = ModelArguments(checkpoint_dir="path_to_lora_checkpoint")
    model, tokenizer = load_pretrained(model_args)
    model.base_model.model.save_pretrained(path_to_save_model)
    tokenizer.save_pretrained(path_to_save_model) # 保存 Tokenizer