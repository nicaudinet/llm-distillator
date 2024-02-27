from enum import Enum
from pathlib import Path

import ollama
from transformers import (AutoModelForCausalLM, AutoTokenizer, Conversation,
                          ConversationalPipeline)


class RunMode(Enum):
    ALVIS = "alvis"
    OLLAMA = "ollama"


class LLMRunner:
    """
    Run Mistral
    """

    def __init__(self, runmode: RunMode):
        self.runmode = runmode

    def run(self, prompt: str) -> str:
        """
        Run Mistral 7B with a prompt
        """
        print(".")
        if self.runmode == RunMode.ALVIS:
            return self._run_alvis_pipeline(prompt)
        else:
            return self._run_ollama(prompt)

    def _run_ollama(self, prompt: str) -> str:
        """
        Run Mistral 7B locally using Ollama with a prompt
        """
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def _run_alvis(self, prompt: str) -> str:
        """
        Run Mistral 7B on my Alvis project with a prompt
        """
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        device = "cuda"
        model_filepath = ()
        model = AutoModelForCausalLM.from_pretrained(model_filepath)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        encodends = tokenizer(prompt, return_tensors="pt")
        model.to(device)
        model_inputs = encodends.to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=200)
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0]

    def _run_alvis_pipeline(self, prompt: str) -> str:
        in_dir = Path("/mimer/NOBACKUP/groups/ci-nlp-alvis/")
        model_path = in_dir / Path("models/Mistral-7B-Instruct-v0.2")
        tokenizer_path = in_dir / Path("tokenizers/Mistral-7B-Instruct-v0.2")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        distillator = ConversationalPipeline(
            model=model,
            tokenizer=tokenizer,
            batch_size=2,
            framework="pt",
        )
        conversation = Conversation(prompt)
        conversation = distillator(conversation)
        return conversation.messages[-1]["content"]
