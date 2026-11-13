from pathlib import Path
from typing import Final
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ConversationHistory:
    MAX_HISTORY_MESSAGES: Final[int] = 10
    START_HISTORY_MESSAGES: Final[int] = 2
    END_HISTORY_MESSAGES: Final[int] = 6

    def __init__(self, system_prompt_path: str = "system_prompt.txt") -> None:
        self._system_prompt: str = self._load_system_prompt(system_prompt_path)
        self._messages: list[dict[str, str]] = []

    def _load_system_prompt(self, path: str) -> str:
        prompt_file = Path(path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"Файл системної інструкції не знайдено: {path}")
        return prompt_file.read_text(encoding="utf-8").strip()

    def add_request(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})
        self._trim_history()

    def add_response(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        self._trim_history()

    def _trim_history(self) -> None:
        if len(self._messages) > self.MAX_HISTORY_MESSAGES:
            start_part = self._messages[: self.START_HISTORY_MESSAGES]
            end_part = self._messages[-self.END_HISTORY_MESSAGES :]
            self._messages = start_part + end_part

    def get_messages(self) -> list[dict[str, str]]:
        system_message = {"role": "system", "content": self._system_prompt}
        return [system_message] + self._messages


class Bot:
    DEFAULT_MODEL_NAME: Final[str] = "microsoft/Phi-4-mini-instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        system_prompt_path: str = "system_prompt.txt",
    ) -> None:
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = ConversationHistory(system_prompt_path)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                merged_messages = [
                    {
                        "role": "user",
                        "content": f"{messages[0]['content']}\n\n{messages[1]['content']}",
                    }
                ] + messages[2:]
                return self.tokenizer.apply_chat_template(
                    merged_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        dialogue_lines = [f"Інструкція: {messages[0]['content']}"]
        for msg in messages[1:]:
            prefix = "Користувач" if msg["role"] == "user" else "Асистент"
            dialogue_lines.append(f"{prefix}: {msg['content']}")
        dialogue_lines.append("Асистент:")
        return "\n".join(dialogue_lines)

    def generate_response(self, user_text: str) -> str:
        self.history.add_request(user_text)
        messages = self.history.get_messages()

        prompt_text = self._format_prompt(messages)

        model_inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)
        input_length = model_inputs.input_ids.shape[1]

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=pad_id,
            )

        new_tokens = generated_ids[0][input_length:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not getattr(self.tokenizer, "chat_template", None):
            if "Користувач:" in response_text:
                response_text = response_text.split("Користувач:")[0].strip()

        self.history.add_response(response_text)
        return response_text