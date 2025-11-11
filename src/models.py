# models.py
"""
Módulo para carregar e executar modelos do Groq e Google Gemini via API.
Suporta múltiplos modelos facilmente.
"""
import warnings
import logging
import os
from groq import Groq
import google.generativeai as genai
from .config import get_config

# Suprimir avisos desnecessários
warnings.filterwarnings("ignore")
logging.getLogger("groq").setLevel(logging.ERROR)

# Carregar configurações
config = get_config()

# Modelos disponíveis no Groq
GROQ_MODELS = {
    "llama3_8b": "llama-3.1-8b-instant",
    "llama3_70b": "llama-3.3-70b-versatile", 
    "gpt_oss_20b": "openai/gpt-oss-20b",
    "gpt_oss_120b": "openai/gpt-oss-120b",
    "qwen_32b": "qwen/qwen3-32b"
}

# Modelos disponíveis no Google Gemini (apenas os 2 funcionais)
GEMINI_MODELS = {
    "gemini_2_5_flash_lite": "models/gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite": "models/gemini-2.0-flash-lite"
    # Nota: gemini_2_5_flash e gemini_2_5_pro estão sendo bloqueados por filtros de segurança
    # Nota: gemini_1_5_pro tem quota limitada, não incluído por enquanto
}

# Dicionário unificado de todos os modelos
AVAILABLE_MODELS = {**GROQ_MODELS, **GEMINI_MODELS}

class ModelRunner:
    """
    Classe para executar modelos do Groq e Google Gemini via API.
    """
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.model_id = AVAILABLE_MODELS.get(model_name)
        
        if not self.model_id:
            raise ValueError(f"Modelo '{model_name}' não encontrado. Modelos disponíveis: {list(AVAILABLE_MODELS.keys())}")
        
        # Determinar se é modelo Groq ou Gemini
        if model_name in GROQ_MODELS:
            self.provider = "groq"
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError("API key do Groq não encontrada. Configure GROQ_API_KEY no .env")
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                raise ValueError(f"Erro ao inicializar cliente Groq: {e}")
                
        elif model_name in GEMINI_MODELS:
            self.provider = "gemini"
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("API key do Gemini não encontrada. Configure GEMINI_API_KEY no .env")
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_id)
            except Exception as e:
                raise ValueError(f"Erro ao inicializar cliente Gemini: {e}")
        else:
            raise ValueError(f"Provedor não suportado para modelo '{model_name}'")

    def generate(self, prompt, **kwargs):
        """
        Gera resposta para um prompt usando o modelo carregado.
        """
        # Parâmetros padrão das configurações centralizadas
        default_params = config.get_model_params()
        
        # Atualizar com parâmetros customizados
        default_params.update(kwargs)
        
        try:
            if self.provider == "groq":
                # Fazer chamada para a API Groq
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=default_params["max_tokens"],
                    temperature=default_params["temperature"],
                    top_p=default_params["top_p"],
                    stream=default_params["stream"]
                )
                
                # Extrair a resposta
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content and content.strip():
                        return content.strip()
                    else:
                        return f"[ERRO]: Resposta vazia do modelo {self.model_name}"
                else:
                    return f"[ERRO]: Nenhuma resposta do modelo {self.model_name}"
                    
            elif self.provider == "gemini":
                # Fazer chamada para a API Gemini
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=default_params["max_tokens"],
                    temperature=default_params["temperature"],
                    top_p=default_params["top_p"]
                )
                
                # Usar configuração padrão (sem safety_settings personalizados)
                # Baseado na documentação: https://ai.google.dev/gemini-api/docs/text-generation?hl=pt-br
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Extrair a resposta diretamente
                try:
                    if response.text and response.text.strip():
                        return response.text.strip()
                    else:
                        # Verificar se há candidatos e qual o motivo do bloqueio
                        if response.candidates:
                            candidate = response.candidates[0]
                            if candidate.finish_reason == 2:  # SAFETY
                                return f"[ERRO]: Conteúdo bloqueado por filtros de segurança para {self.model_name}"
                            elif candidate.finish_reason == 3:  # RECITATION
                                return f"[ERRO]: Conteúdo bloqueado por recitação para {self.model_name}"
                            elif candidate.finish_reason == 4:  # OTHER
                                return f"[ERRO]: Resposta bloqueada por outros motivos para {self.model_name}"
                            else:
                                return f"[ERRO]: Resposta vazia do modelo {self.model_name} (finish_reason: {candidate.finish_reason})"
                        else:
                            return f"[ERRO]: Nenhuma resposta do modelo {self.model_name}"
                except Exception as text_error:
                    # Se response.text falhar, verificar candidatos diretamente
                    if response.candidates:
                        candidate = response.candidates[0]
                        if candidate.finish_reason == 2:  # SAFETY
                            return f"[ERRO]: Conteúdo bloqueado por filtros de segurança para {self.model_name}"
                        elif candidate.finish_reason == 3:  # RECITATION
                            return f"[ERRO]: Conteúdo bloqueado por recitação para {self.model_name}"
                        elif candidate.finish_reason == 4:  # OTHER
                            return f"[ERRO]: Resposta bloqueada por outros motivos para {self.model_name}"
                        else:
                            return f"[ERRO]: Resposta vazia do modelo {self.model_name} (finish_reason: {candidate.finish_reason})"
                    else:
                        return f"[ERRO]: Erro ao acessar resposta do modelo {self.model_name}: {str(text_error)}"
            else:
                return f"[ERRO]: Provedor não suportado para {self.model_name}"
                
        except Exception as e:
            error_msg = str(e)
            
            # Tratamento específico de erros
            if "rate_limit" in error_msg.lower() or "quota" in error_msg.lower():
                return f"[ERRO]: Rate limit ou quota excedida para {self.model_name}"
            elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                return f"[ERRO]: Problema de autenticação para {self.model_name}"
            elif "not_found" in error_msg.lower() or "404" in error_msg.lower():
                return f"[ERRO]: Modelo não encontrado: {self.model_name}"
            elif "timeout" in error_msg.lower():
                return f"[ERRO]: Timeout na requisição para {self.model_name}"
            elif "context_length" in error_msg.lower() or "token" in error_msg.lower():
                return f"[ERRO]: Prompt muito longo para {self.model_name}"
            elif "safety" in error_msg.lower() or "blocked" in error_msg.lower():
                return f"[ERRO]: Conteúdo bloqueado por filtros de segurança para {self.model_name}"
            elif "permission" in error_msg.lower():
                return f"[ERRO]: Sem permissão para acessar {self.model_name}"
            elif not error_msg.strip():
                return f"[ERRO]: Erro desconhecido com {self.model_name}"
            else:
                return f"[ERRO]: {error_msg} (modelo: {self.model_name})"

    def get_model_info(self):
        """
        Retorna informações sobre o modelo atual.
        """
        try:
            if self.provider == "groq":
                # Tentar obter informações do modelo via API Groq
                models_response = self.client.models.list()
                
                for model in models_response.data:
                    if model.id == self.model_id:
                        return {
                            "id": model.id,
                            "object": model.object,
                            "created": model.created,
                            "owned_by": model.owned_by,
                            "provider": "groq"
                        }
                
                return {"id": self.model_id, "status": "Modelo não encontrado na lista", "provider": "groq"}
                
            elif self.provider == "gemini":
                # Informações básicas do Gemini
                return {
                    "id": self.model_id,
                    "object": "model",
                    "created": None,
                    "owned_by": "google",
                    "provider": "gemini"
                }
            else:
                return {"id": self.model_id, "status": "Provedor não suportado", "provider": "unknown"}
            
        except Exception as e:
            return {"id": self.model_id, "error": str(e), "provider": self.provider} 