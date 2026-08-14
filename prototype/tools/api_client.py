import os
import time
import logging
from openai import OpenAI
from anthropic import Anthropic
import openai

from prototype.config import api_key

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.ERROR)

MAX_RETRIES = 5

class BaseClient:
    def __init__(self, key_path, model, url=None, provider=None):
        self.key = api_key(provider, key_path) if provider else "EMPTY"
        self.model = model
        self.url = url
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def reset_token_usage(self):
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_token_usage(self) -> dict:
        return dict(self._token_usage)

    def obtain_response(
        self,
        prompt: str = None,
        messages: list = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        response = None
        num_attempts = 0
        while response is None and num_attempts < MAX_RETRIES:
            try:
                response = self.send_request(prompt=prompt, messages=messages, max_tokens=max_tokens, temperature=temperature)
            except openai.BadRequestError as e:
                error_msg = str(e)
                # if "longer than the maximum model length" in error_msg or "max_model_len" in error_msg:
                #     print(f"[Error] Context too long, cannot retry: {e}")
                #     raise
                num_attempts += 1
                if num_attempts >= MAX_RETRIES:
                    raise
                print(f"Attempt {num_attempts} failed, trying again after 5 seconds...")
                time.sleep(5)
            except Exception as e:
                print(e)
                num_attempts += 1
                if num_attempts >= MAX_RETRIES:
                    raise
                print(f"Attempt {num_attempts} failed, trying again after 5 seconds...")
                time.sleep(5)
        text, usage = response
        for k in self._token_usage:
            self._token_usage[k] += usage[k]
        return text

    def obtain_response_with_tools(self, prompt=None, messages=None, tools=None, tool_choice="auto"):
        response = None
        num_attempts = 0
        while response is None and num_attempts < MAX_RETRIES:
            try:
                response = self.send_request_with_tools(prompt=prompt, messages=messages, tools=tools, tool_choice=tool_choice)
            except Exception as e:
                print(e)
                num_attempts += 1
                if num_attempts >= MAX_RETRIES:
                    raise
                print(f"Attempt {num_attempts} failed, trying again after 5 seconds...")
                time.sleep(5)
        return response

    def send_request(self, prompt=None, messages=None, max_tokens=512, temperature=0.0):
        raise NotImplementedError("send_request method must be implemented by subclasses.")

    def send_request_with_tools(self, prompt=None, messages=None, tools=None, tool_choice="auto"):
        raise NotImplementedError("send_request_with_tools method must be implemented by subclasses.")

# Fixme: Seperate the thinking mode

class OpenAIClient(BaseClient):
    def __init__(self, key_path, model, url=None, think_mode=None, provider=None):
        super().__init__(key_path, model, url, provider=provider)
        self.think_mode = think_mode
        if self.url:
            self.client = OpenAI(api_key=self.key, base_url=self.url)
        else:
            self.client = OpenAI(api_key=self.key)

    def send_request(self, prompt=None, messages=None, max_tokens=512, temperature=0.0):

        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        if self.think_mode == False:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        else:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        usage = {"prompt_tokens": r.usage.prompt_tokens if r.usage else 0,
                 "completion_tokens": r.usage.completion_tokens if r.usage else 0,
                 "total_tokens": r.usage.total_tokens if r.usage else 0}
        return r.choices[0].message.content, usage

    def send_request_with_tools(self, prompt=None, messages=None, tools=None, tool_choice="auto"):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response

    def get_embedding(self, text, model):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding

class AnthropicClient(BaseClient):
    def __init__(self, key_path, model):
        super().__init__(key_path, model, provider="anthropic")
        self.client = Anthropic(api_key=self.key)

    def send_request(self, prompt=None, messages=None, max_tokens=512, temperature=0.0):

        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        r = self.client.messages.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        usage = {"prompt_tokens": r.usage.input_tokens,
                 "completion_tokens": r.usage.output_tokens,
                 "total_tokens": r.usage.input_tokens + r.usage.output_tokens}
        return r.content[0].text, usage

class QwenOmniClient(BaseClient):
    def __init__(self, key_path, model, url=None, debug=False, download_dir=None):
        super().__init__(key_path, model, url)
        try:
            from transformers import (
                GenerationConfig,
                Qwen2_5OmniForConditionalGeneration,
                Qwen2_5OmniProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "Qwen Omni support requires the optional 'omni' dependencies."
            ) from exc
        self._generation_config_cls = GenerationConfig
        self._process_mm_info = __import__(
            "qwen_omni_utils", fromlist=["process_mm_info"]
        ).process_mm_info
        self.llm = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model)

    def send_request(self, prompt=None, messages=None, max_tokens=512, temperature=0.0):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generation_config = self._generation_config_cls(pad_token_id=151643, bos_token_id=151644, eos_token_id=151645)
        USE_AUDIO_IN_VIDEO = True
        try:
            audios, images, videos = self._process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        except Exception as e:
            if "Video must has audio track" in str(e):
                USE_AUDIO_IN_VIDEO = False
                audios, images, videos = self._process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            else:
                raise e
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(self.llm.device).to(self.llm.dtype)
        text_ids, _ = self.llm.generate(**inputs, generation_config=generation_config, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=4096, temperature=0.0)
        text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return text, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

class APIClient():
    def __init__(self, api, key_path, model, embedding_model, debug=False, download_dir=None, ip_address=None, think_mode=None):
        self.api = api
        self.model = model
        self.embedding_model = embedding_model
        link_url = os.getenv("PYRAVID_LINK_BASE_URL")
        selection_url = os.getenv("PYRAVID_SELECTION_BASE_URL")
        answer_url = os.getenv("PYRAVID_ANSWER_BASE_URL")
        match api:
            case "openai":
                self.client = OpenAIClient(key_path, model, provider="openai")
            case "anthropic":
                self.client = AnthropicClient(key_path, model)
            case "qwen":
                self.client = OpenAIClient(None, model, url=selection_url, think_mode=think_mode)
            case "gemini":
                self.client = OpenAIClient(key_path, model, url='https://generativelanguage.googleapis.com/v1beta/openai/', provider="gemini")
            case "qwen-omni":
                self.client = QwenOmniClient(None, model, debug=debug, download_dir=download_dir)
            case "qwen-server-link-model":
                self.client = OpenAIClient(None, model, url=link_url, think_mode=think_mode)
            case "qwen-server-selection-model":
                url = ip_address if ip_address else selection_url
                self.client = OpenAIClient(None, model, url=url, think_mode=think_mode)
            case "qwen-server-answer-model":
                url = ip_address if ip_address else answer_url
                self.client = OpenAIClient(None, model, url=url, think_mode=think_mode)
            case _:
                raise ValueError(f"API {api} not supported, custom implementation required.")

    def reset_token_usage(self):
        self.client.reset_token_usage()

    def get_token_usage(self) -> dict:
        return self.client.get_token_usage()

    def obtain_response(
        self,
        prompt: str = None,
        messages: list = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):

        return self.client.obtain_response(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def obtain_response_with_tools(self, prompt=None, messages=None, tools=None, tool_choice="auto"):
        return self.client.obtain_response_with_tools(
            prompt=prompt,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

    def obtain_embedding(self, input):
        return self.client.get_embedding(input, self.embedding_model)
