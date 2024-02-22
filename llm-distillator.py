from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = "cuda"
    model_filepath = "/mimer/NOBACKUP/groups/ci-nlp-alvis/models/Mistral-7B-Instruct-v0.2"

    model = AutoModelForCausalLM.from_pretrained(model_filepath)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    encodends = tokenizer("The capital of France is", return_tensors="pt")

    model.to(device)
    model_inputs = encodends.to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=200)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print(decoded[0])
