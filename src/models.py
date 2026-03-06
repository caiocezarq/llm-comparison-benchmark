# models.py
"""
M?dulo para carregar e executar modelos do Groq e Google Gemini via API.
Suporta m?ltiplos modelos facilmente.
"""
import warnings
import logging
import os
import time
import random
from groq import Groq
import google.generativeai as genai
from .config import get_config

# Suprimir avisos desnecess?rios
warnings.filterwarnings("ignore")
logging.getLogger("groq").setLevel(logging.ERROR)

# Carregar configura??es
config = get_config()

# Modelos dispon?veis no Groq
GROQ_MODELS = {
    "llama3_8b": "llama-3.1-8b-instant",
    "llama3_70b": "llama-3.3-70b-versatile",
    "gpt_oss_20b": "openai/gpt-oss-20b",
    "gpt_oss_120b": "openai/gpt-oss-120b",
    "qwen_32b": "qwen/qwen3-32b"
}

# Modelos dispon?veis no Google Gemini
GEMINI_MODELS = {
    "gemini_2_5_flash_lite": "models/gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite": "models/gemini-2.0-flash-lite"
}

# Dicion?rio unificado de todos os modelos
AVAILABLE_MODELS = {**GROQ_MODELS, **GEMINI_MODELS}


class ModelRunner:
    """
    Classe para executar modelos do Groq e Google Gemini via API.
    """

    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.model_id = AVAILABLE_MODELS.get(model_name)

        if not self.model_id:
            raise ValueError(
                f"Modelo '{model_name}' n?o encontrado. Modelos dispon?veis: {list(AVAILABLE_MODELS.keys())}"
            )

        if model_name in GROQ_MODELS:
            self.provider = "groq"
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError("API key do Groq n?o encontrada. Configure GROQ_API_KEY no .env")
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                raise ValueError(f"Erro ao inicializar cliente Groq: {e}")

        elif model_name in GEMINI_MODELS:
            self.provider = "gemini"
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("API key do Gemini n?o encontrada. Configure GEMINI_API_KEY no .env")
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_id)
            except Exception as e:
                raise ValueError(f"Erro ao inicializar cliente Gemini: {e}")
        else:
            raise ValueError(f"Provedor n?o suportado para modelo '{model_name}'")

    def _is_retryable_error(self, message: str) -> bool:
        """Define erros transit?rios pass?veis de retry."""
        if not message:
            return False
        msg = message.lower()
        retry_patterns = [
            "rate limit",
            "rate_limit",
            "quota",
            "timeout",
            "timed out",
            "429",
            "503",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "network",
        ]
        return any(p in msg for p in retry_patterns)

    def _format_error(self, error_msg: str) -> str:
        """Normaliza mensagens de erro para o formato do pipeline."""
        msg = (error_msg or "").lower()
        if "rate_limit" in msg or "rate limit" in msg or "quota" in msg:
            return f"[ERRO]: Rate limit ou quota excedida para {self.model_name}"
        if "authentication" in msg or "unauthorized" in msg:
            return f"[ERRO]: Problema de autentica??o para {self.model_name}"
        if "not_found" in msg or "404" in msg:
            return f"[ERRO]: Modelo n?o encontrado: {self.model_name}"
        if "timeout" in msg:
            return f"[ERRO]: Timeout na requisi??o para {self.model_name}"
        if "context_length" in msg or "token" in msg:
            return f"[ERRO]: Prompt muito longo para {self.model_name}"
        if "safety" in msg or "blocked" in msg:
            return f"[ERRO]: Conte?do bloqueado por filtros de seguran?a para {self.model_name}"
        if "permission" in msg:
            return f"[ERRO]: Sem permiss?o para acessar {self.model_name}"
        if not (error_msg or "").strip():
            return f"[ERRO]: Erro desconhecido com {self.model_name}"
        return f"[ERRO]: {error_msg} (modelo: {self.model_name})"

    def _generate_once(self, prompt, **kwargs):
        """Realiza uma tentativa ?nica de gera??o."""
        default_params = config.get_model_params()
        default_params.update(kwargs)

        try:
            if self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=default_params["max_tokens"],
                    temperature=default_params["temperature"],
                    top_p=default_params["top_p"],
                    stream=default_params["stream"],
                )

                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content and content.strip():
                        return content.strip()
                    return f"[ERRO]: Resposta vazia do modelo {self.model_name}"
                return f"[ERRO]: Nenhuma resposta do modelo {self.model_name}"

            if self.provider == "gemini":
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=default_params["max_tokens"],
                    temperature=default_params["temperature"],
                    top_p=default_params["top_p"],
                )

                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )

                try:
                    if response.text and response.text.strip():
                        return response.text.strip()
                    if response.candidates:
                        candidate = response.candidates[0]
                        if candidate.finish_reason == 2:
                            return f"[ERRO]: Conte?do bloqueado por filtros de seguran?a para {self.model_name}"
                        if candidate.finish_reason == 3:
                            return f"[ERRO]: Conte?do bloqueado por recita??o para {self.model_name}"
                        if candidate.finish_reason == 4:
                            return f"[ERRO]: Resposta bloqueada por outros motivos para {self.model_name}"
                        return f"[ERRO]: Resposta vazia do modelo {self.model_name} (finish_reason: {candidate.finish_reason})"
                    return f"[ERRO]: Nenhuma resposta do modelo {self.model_name}"
                except Exception as text_error:
                    if response.candidates:
                        candidate = response.candidates[0]
                        if candidate.finish_reason == 2:
                            return f"[ERRO]: Conte?do bloqueado por filtros de seguran?a para {self.model_name}"
                        if candidate.finish_reason == 3:
                            return f"[ERRO]: Conte?do bloqueado por recita??o para {self.model_name}"
                        if candidate.finish_reason == 4:
                            return f"[ERRO]: Resposta bloqueada por outros motivos para {self.model_name}"
                        return f"[ERRO]: Resposta vazia do modelo {self.model_name} (finish_reason: {candidate.finish_reason})"
                    return f"[ERRO]: Erro ao acessar resposta do modelo {self.model_name}: {str(text_error)}"

            return f"[ERRO]: Provedor n?o suportado para {self.model_name}"

        except Exception as e:
            return self._format_error(str(e))

    def generate(self, prompt, **kwargs):
        """Gera resposta com retry/backoff para erros transit?rios."""
        max_retries = int(getattr(config, "MAX_RETRIES", 0))
        retry_delay = float(getattr(config, "RETRY_DELAY", 1))
        last_result = None

        for attempt in range(max_retries + 1):
            result = self._generate_once(prompt, **kwargs)
            last_result = result

            if not (isinstance(result, str) and result.startswith("[ERRO]")):
                return result

            if attempt >= max_retries or not self._is_retryable_error(result):
                return result

            sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(sleep_time)

        return last_result or f"[ERRO]: Falha desconhecida em {self.model_name}"

    def get_model_info(self):
        """Retorna informa??es sobre o modelo atual."""
        try:
            if self.provider == "groq":
                models_response = self.client.models.list()
                for model in models_response.data:
                    if model.id == self.model_id:
                        return {
                            "id": model.id,
                            "object": model.object,
                            "created": model.created,
                            "owned_by": model.owned_by,
                            "provider": "groq",
                        }
                return {"id": self.model_id, "status": "Modelo n?o encontrado na lista", "provider": "groq"}

            if self.provider == "gemini":
                return {
                    "id": self.model_id,
                    "object": "model",
                    "created": None,
                    "owned_by": "google",
                    "provider": "gemini",
                }

            return {"id": self.model_id, "status": "Provedor n?o suportado", "provider": "unknown"}

        except Exception as e:
            return {"id": self.model_id, "error": str(e), "provider": self.provider}
