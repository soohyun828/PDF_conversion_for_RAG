from huggingface_hub import hf_hub_download


# download model
model_name_or_path = "heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF" # repo id
# 4bit
model_basename = "ggml-model-Q4_K_M.gguf" # file name

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
print(model_path)