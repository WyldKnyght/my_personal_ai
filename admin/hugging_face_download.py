from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TheBloke/Llama-2-7b-Chat-GGUF"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
