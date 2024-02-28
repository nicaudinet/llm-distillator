from enum import Enum
from pathlib import Path

import ollama
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Conversation,
    ConversationalPipeline,
)


class RunMode(Enum):
    ALVIS = "alvis"
    OLLAMA = "ollama"


class LLMRunner:
    """
    Run Mistral
    """

    def __init__(self, runmode: RunMode):
        self.runmode = runmode

    def run(self, prompts: list[str]) -> list[str]:
        """
        Run Mistral 7B with a prompt
        """
        print(".")
        if self.runmode == RunMode.ALVIS:
            return self._run_alvis_pipeline(prompts)
        else:
            return self._run_ollama(prompts)

    def _run_ollama(self, prompts: list[str]) -> list[str]:
        """
        Run Mistral 7B locally using Ollama with a prompt
        """
        responses = []
        for prompt in prompts:
            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
            )
            responses.append(response["message"]["content"])
        return responses

    def _run_alvis_pipeline(self, prompts: list[str]) -> list[str]:
        in_dir = Path("/mimer/NOBACKUP/groups/ci-nlp-alvis/")
        model_path = in_dir / Path("models/Mistral-7B-Instruct-v0.2")
        tokenizer_path = in_dir / Path("tokenizers/Mistral-7B-Instruct-v0.2")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        distillator = ConversationalPipeline(
            model=model,
            tokenizer=tokenizer,
            batch_size=2,
            framework="pt",
        )
        responses = []
        for prompt in prompts:
            conversation = Conversation(prompt)
            conversation = distillator(conversation)
            response = conversation.messages[-1]["content"]
            responses.append(response)
        return responses
