# config.py
"""
Configurações centralizadas do projeto.
Edite este arquivo para ajustar as configurações conforme necessário.
"""

from typing import Dict, Any


class Config:
    """
    Configurações centralizadas do projeto.
    Edite os valores abaixo conforme sua necessidade.
    """
    
    # =============================================================================
    # CONFIGURAÇÕES DE EXECUÇÃO
    # =============================================================================
    # Número de execuções da pipeline (mínimo 1)
    NUMERO_EXECUCOES = 3
    
    # Timeout em segundos entre execuções (para evitar problemas com APIs)
    TIMEOUT_ENTRE_EXECUCOES = 30
    
    # Timeout entre perguntas durante a execução da pipeline (em segundos)
    TIMEOUT_ENTRE_PERGUNTAS = 3
    
    
    # =============================================================================
    # CONFIGURAÇÕES DE MODELOS
    # =============================================================================
    # Parâmetros padrão para geração de texto
    MAX_TOKENS = 250
    TEMPERATURE = 0.7
    TOP_P = 1.0
    STREAM = False
    
    # =============================================================================
    # CONFIGURAÇÕES DE PASTAS E ARQUIVOS
    # =============================================================================
    PASTA_RESULTADOS = "results"
    PREFIXO_EXECUCAO = "resultado"
    PASTA_TESTS = "tests"
    
    # =============================================================================
    # CONFIGURAÇÕES DE LOGGING
    # =============================================================================
    NIVEL_LOG = "INFO"
    FORMATO_LOG = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # =============================================================================
    # CONFIGURAÇÕES DE MÉTRICAS
    # =============================================================================
    # Configurações para BERTScore
    BERT_SCORE_LANG = "en"
    BERT_SCORE_MODEL_TYPE = "roberta-large"
    BERT_SCORE_USE_FAST_TOKENIZER = True
    
    # =============================================================================
    # CONFIGURAÇÕES DE BENCHMARKS
    # =============================================================================
    # Incluir benchmarks padronizados (MMLU, HellaSwag)
    INCLUDE_BENCHMARKS = True
    
    # Pasta e arquivo de prompts de benchmarks
    BENCHMARKS_FOLDER = "prompts"
    BENCHMARKS_FILE = "benchmarks.json"
    
    # Pasta e arquivo de prompts padrão
    PROMPTS_FOLDER = "prompts"
    PROMPTS_FILE = "prompts.json"
    
    # =============================================================================
    # CONFIGURAÇÕES DE ENCODING
    # =============================================================================
    ENCODING_CSV = "utf-8-sig"
    ENCODING_JSON = "utf-8"
    ENCODING_TXT = "utf-8"
    
    # =============================================================================
    # CONFIGURAÇÕES DE API
    # =============================================================================
    # Timeout para requisições de API (em segundos)
    API_TIMEOUT = 60
    
    # Número máximo de tentativas para requisições
    MAX_RETRIES = 3
    
    # Delay entre tentativas (em segundos)
    RETRY_DELAY = 5
    
    @classmethod
    def get_model_params(cls) -> Dict[str, Any]:
        """
        Retorna parâmetros padrão para modelos.
        
        Returns:
            Dict[str, Any]: Parâmetros de configuração dos modelos
        """
        return {
            "max_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE,
            "top_p": cls.TOP_P,
            "stream": cls.STREAM
        }
    
    @classmethod
    def get_bertscore_params(cls) -> Dict[str, Any]:
        """
        Retorna parâmetros para BERTScore.
        
        Returns:
            Dict[str, Any]: Parâmetros de configuração do BERTScore
        """
        return {
            "lang": cls.BERT_SCORE_LANG,
            "model_type": cls.BERT_SCORE_MODEL_TYPE,
            "use_fast_tokenizer": cls.BERT_SCORE_USE_FAST_TOKENIZER
        }
    
    @classmethod
    def get_encoding_config(cls) -> Dict[str, str]:
        """
        Retorna configurações de encoding.
        
        Returns:
            Dict[str, str]: Configurações de encoding para diferentes tipos de arquivo
        """
        return {
            "csv": cls.ENCODING_CSV,
            "json": cls.ENCODING_JSON,
            "txt": cls.ENCODING_TXT
        }
    
    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """
        Retorna configurações de API.
        
        Returns:
            Dict[str, Any]: Configurações para requisições de API
        """
        return {
            "timeout": cls.API_TIMEOUT,
            "max_retries": cls.MAX_RETRIES,
            "retry_delay": cls.RETRY_DELAY
        }
    
    @classmethod
    def get_benchmarks_config(cls) -> Dict[str, Any]:
        """
        Retorna configurações de benchmarks.
        
        Returns:
            Dict[str, Any]: Configurações para benchmarks
        """
        return {
            "include_benchmarks": cls.INCLUDE_BENCHMARKS,
            "benchmarks_folder": cls.BENCHMARKS_FOLDER,
            "benchmarks_file": cls.BENCHMARKS_FILE,
            "prompts_folder": cls.PROMPTS_FOLDER,
            "prompts_file": cls.PROMPTS_FILE
        }


class ConfigValidator:
    """
    Classe para validar configurações.
    """
    
    @staticmethod
    def validate_execution_config() -> None:
        """
        Valida configurações de execução.
        
        Raises:
            ValueError: Se alguma configuração for inválida
        """
        if Config.NUMERO_EXECUCOES < 1:
            raise ValueError("NUMERO_EXECUCOES deve ser >= 1")
        
        if Config.TIMEOUT_ENTRE_EXECUCOES < 0:
            raise ValueError("TIMEOUT_ENTRE_EXECUCOES deve ser >= 0")
        
    
    @staticmethod
    def validate_model_config() -> None:
        """
        Valida configurações de modelos.
        
        Raises:
            ValueError: Se alguma configuração for inválida
        """
        if Config.MAX_TOKENS < 1:
            raise ValueError("MAX_TOKENS deve ser >= 1")
        
        if not 0 <= Config.TEMPERATURE <= 2:
            raise ValueError("TEMPERATURE deve estar entre 0 e 2")
        
        if not 0 <= Config.TOP_P <= 1:
            raise ValueError("TOP_P deve estar entre 0 e 1")
    
    @staticmethod
    def validate_api_config() -> None:
        """
        Valida configurações de API.
        
        Raises:
            ValueError: Se alguma configuração for inválida
        """
        if Config.API_TIMEOUT < 1:
            raise ValueError("API_TIMEOUT deve ser >= 1")
        
        if Config.MAX_RETRIES < 0:
            raise ValueError("MAX_RETRIES deve ser >= 0")
        
        if Config.RETRY_DELAY < 0:
            raise ValueError("RETRY_DELAY deve ser >= 0")
        
        if Config.TIMEOUT_ENTRE_PERGUNTAS < 0:
            raise ValueError("TIMEOUT_ENTRE_PERGUNTAS deve ser >= 0")
    
    @staticmethod
    def validate_metrics_config() -> None:
        """
        Valida configurações de métricas.
        
        Raises:
            ValueError: Se alguma configuração for inválida
        """
        if not isinstance(Config.BERT_SCORE_LANG, str) or not Config.BERT_SCORE_LANG:
            raise ValueError("BERT_SCORE_LANG deve ser uma string não vazia")
        
        if not isinstance(Config.BERT_SCORE_MODEL_TYPE, str) or not Config.BERT_SCORE_MODEL_TYPE:
            raise ValueError("BERT_SCORE_MODEL_TYPE deve ser uma string não vazia")
        
        if not isinstance(Config.BERT_SCORE_USE_FAST_TOKENIZER, bool):
            raise ValueError("BERT_SCORE_USE_FAST_TOKENIZER deve ser um booleano")
    
    @classmethod
    def validate_all(cls) -> None:
        """
        Valida todas as configurações.
        
        Raises:
            ValueError: Se alguma configuração for inválida
        """
        cls.validate_execution_config()
        cls.validate_model_config()
        cls.validate_api_config()
        cls.validate_metrics_config()


def get_config() -> Config:
    """
    Retorna a configuração centralizada.
    
    Returns:
        Config: Instância de configuração
    """
    return Config()
