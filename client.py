import requests
from typing import List, Dict
from indic_assistant.utils.config import Config
from indic_assistant.utils.logger import logger
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class LLMClient:

    def __init__(self):
        self.provider = Config.LLM_PROVIDER.strip().lower()
        self.model = Config.LLM_MODEL.strip()
        self.temperature = Config.LLM_TEMPERATURE
        self.gemini_key = Config.GEMINI_API_KEY
        self.openai_key = Config.OPENAI_API_KEY
        self.base_url = Config.LLM_BASE_URL
        self.hf_token = Config.HUGGINGFACE_TOKEN
        
        # Local transformers model references
        self._local_tokenizer = None
        self._local_model = None

        # Normalize provider names
        if self.provider in ["gemini"]:
            self.provider = "gemini"
        elif self.provider in ["openai"]:
            self.provider = "openai"
        elif self.provider in ["ollama"]:
            self.provider = "ollama"
        elif self.provider in ["openai-compatible", "openai_compatible"]:
            self.provider = "openai-compatible"
        elif self.provider in ["huggingface", "hf", "transformers-local", "transformers"]:
            self.provider = "transformers-local"
            # Map Ollama tags / shortnames to official Hugging Face Repository IDs
            model_lower = self.model.lower()
            if "qwen" in model_lower:
                if "qwen3" in model_lower or "qwen-3" in model_lower or "1.7b" in model_lower:
                    self.model = "Qwen/Qwen2.5-1.5B-Instruct"
                    logger.info(f"Auto-mapped Qwen model to Hugging Face repo: {self.model}")
                elif "7b" in model_lower:
                    self.model = "Qwen/Qwen2.5-7B-Instruct"
                    logger.info(f"Auto-mapped Qwen model to Hugging Face repo: {self.model}")
                else:
                    self.model = "Qwen/Qwen2.5-1.5B-Instruct"
                    logger.info(f"Auto-mapped Qwen model to Hugging Face repo: {self.model}")
        else:
            self.provider = "mockup"

        logger.info(f"Initializing LLMClient using provider: {self.provider} and model: {self.model}")
        
        # Configure endpoints & validation
        if self.provider == "gemini" and not self.gemini_key:
            logger.warning("GEMINI_API_KEY not found. Falling back to mockup provider.")
            self.provider = "mockup"
        elif self.provider == "openai" and not self.openai_key:
            logger.warning("OPENAI_API_KEY not found. Falling back to mockup provider.")
            self.provider = "mockup"
        elif self.provider == "ollama" and not self.base_url:
            self.base_url = "http://localhost:11434/v1"
            logger.info(f"Defaulting Ollama base URL to {self.base_url}")
        elif self.provider == "openai-compatible" and not self.base_url:
            self.base_url = "http://localhost:8000/v1"
            logger.info(f"Defaulting OpenAI-compatible base URL to {self.base_url}")

    def query(self, prompt: str, history: List[Dict[str, str]] = None) -> str:
        """
        Sends the query + history to the configured LLM provider.
        history format: list of {"role": "user"/"assistant", "content": "text"}
        """
        if self.provider == "gemini":
            return self._query_gemini(prompt, history or [])
        elif self.provider == "openai":
            return self._query_openai(prompt, history or [])
        elif self.provider in ["ollama", "openai-compatible"]:
            return self._query_openai_compatible(prompt, history or [])
        elif self.provider == "transformers-local":
            return self._query_transformers_local(prompt, history or [])
        else:
            return self._query_mockup(prompt, history or [])

    def _query_gemini(self, prompt: str, history: List[Dict[str, str]]) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.gemini_key}"
        
        # Build Gemini payload
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        
        # Append the new prompt
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature
            }
        }
        
        try:
            logger.info("Sending request to Gemini API...")
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            res_json = response.json()
            
            text = res_json['candidates'][0]['content']['parts'][0]['text']
            return text.strip()
        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            return self._query_mockup(prompt, history)

    def _query_openai(self, prompt: str, history: List[Dict[str, str]]) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        return self._query_openai_like(url, headers, prompt, history)

    def _query_openai_compatible(self, prompt: str, history: List[Dict[str, str]]) -> str:
        # Standard chat completions endpoint
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        logger.info(f"Routing request to OpenAI-compatible endpoint: {url}")
        return self._query_openai_like(url, headers, prompt, history)

    def _query_openai_like(self, url: str, headers: dict, prompt: str, history: List[Dict[str, str]]) -> str:
        messages = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            res_json = response.json()
            
            text = res_json['choices'][0]['message']['content']
            return text.strip()
        except Exception as e:
            logger.error(f"OpenAI-like endpoint request failed: {e}")
            return self._query_mockup(prompt, history)

    def warmup(self):
        """Warmup client by pre-loading local models during startup."""
        if self.provider == "transformers-local":
            logger.info("Warmup: pre-loading local transformers LLM model...")
            self._load_local_model()

    def _load_local_model(self) -> bool:
        """Loads local model and tokenizer into memory if not already loaded. Returns True if successful."""
        if self._local_model is not None:
            return True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.error("Required libraries (torch, transformers) not installed for local LLM.")
            return False

        logger.info(f"Loading local LLM model '{self.model}' in-process...")
        try:
            # Use CPU/CUDA target settings
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._local_tokenizer = AutoTokenizer.from_pretrained(self.model, token=self.hf_token)
            self._local_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                token=self.hf_token,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                attn_implementation="sdpa" if device == "cuda" else None
            )
            if device == "cpu":
                self._local_model = self._local_model.to("cpu")
            logger.info("Local LLM model loaded successfully.")
            return True
        except Exception as e:
            import traceback
            logger.error(f"Failed to load local model: {e}")
            logger.error(traceback.format_exc())
            return False

    def _query_transformers_local(self, prompt: str, history: List[Dict[str, str]]) -> str:
        """Loads and runs Qwen / LLM locally in the python process using transformers."""
        if not self._load_local_model():
            return self._query_mockup(prompt, history)

        try:
            # Build conversation format for Qwen/Chat templates
            messages = []
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})

            text = self._local_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self._local_tokenizer([text], return_tensors="pt").to(self._local_model.device)

            generated_ids = self._local_model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self._local_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.info(f"Local LLM generated: '{response[:50]}...'")
            return response.strip()
        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            return self._query_mockup(prompt, history)

    def _query_mockup(self, prompt: str, history: List[Dict[str, str]]) -> str:
        logger.info("Using mockup LLM response generator.")
        query_lower = prompt.lower()
        if "hello" in query_lower or "hi" in query_lower:
            return "Hello! I am your AI Voice Assistant. How can I help you today?"
        elif "time" in query_lower:
            import datetime
            now = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current local time is {now}."
        elif "name" in query_lower:
            return "I am the Indic Multilingual AI Voice Assistant."
        else:
            return f"I received your message: '{prompt}'. This is a local mockup response because no LLM API keys were set or the API request timed out."
