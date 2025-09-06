# logger.py
"""
Sistema de logging estruturado para o projeto.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from .config import get_config


class ColoredFormatter(logging.Formatter):
    """
    Formatter colorido para logs no terminal.
    """
    
    # Cores ANSI
    COLORS = {
        'DEBUG': '\033[36m',    # Ciano
        'INFO': '\033[32m',     # Verde
        'WARNING': '\033[33m',  # Amarelo
        'ERROR': '\033[31m',    # Vermelho
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Adicionar cor baseada no nível
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logger(name: str = None, level: str = None) -> logging.Logger:
    """
    Configura e retorna um logger estruturado.
    
    Args:
        name (str): Nome do logger
        level (str): Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Logger configurado
    """
    # Carregar configurações
    config = get_config()
    
    # Configurar nome e nível
    logger_name = name or __name__
    log_level = level or config.NIVEL_LOG
    
    # Criar logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Evitar duplicação de handlers
    if logger.handlers:
        return logger
    
    # Criar pasta de logs se não existir
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Handler para arquivo
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = logs_dir / f"pipeline_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter para arquivo (sem cores)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter para console (com cores)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # Adicionar handlers ao logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Retorna um logger configurado.
    
    Args:
        name (str): Nome do logger
        
    Returns:
        logging.Logger: Logger configurado
    """
    return setup_logger(name)


# Logger principal do projeto
main_logger = get_logger("pipeline_llm")


def log_execution_start(logger: logging.Logger, execucao: int, total: int):
    """
    Log do início de uma execução.
    
    Args:
        logger (logging.Logger): Logger a ser usado
        execucao (int): Número da execução atual
        total (int): Total de execuções
    """
    logger.info(f"🚀 Iniciando execução {execucao}/{total}")


def log_execution_end(logger: logging.Logger, execucao: int, sucesso: bool, tempo: float):
    """
    Log do fim de uma execução.
    
    Args:
        logger (logging.Logger): Logger a ser usado
        execucao (int): Número da execução
        sucesso (bool): Se a execução foi bem-sucedida
        tempo (float): Tempo de execução em segundos
    """
    status = "✅ Sucesso" if sucesso else "❌ Erro"
    logger.info(f"{status} - Execução {execucao} concluída em {tempo:.2f}s")


def log_model_execution(logger: logging.Logger, modelo: str, prompt_num: int, total_prompts: int, sucesso: bool):
    """
    Log da execução de um modelo.
    
    Args:
        logger (logging.Logger): Logger a ser usado
        modelo (str): Nome do modelo
        prompt_num (int): Número do prompt atual
        total_prompts (int): Total de prompts
        sucesso (bool): Se a execução foi bem-sucedida
    """
    status = "✅" if sucesso else "❌"
    logger.debug(f"{status} {modelo} - Prompt {prompt_num}/{total_prompts}")


def log_error(logger: logging.Logger, erro: Exception, contexto: str = ""):
    """
    Log de erro com contexto.
    
    Args:
        logger (logging.Logger): Logger a ser usado
        erro (Exception): Exceção ocorrida
        contexto (str): Contexto adicional
    """
    contexto_str = f" ({contexto})" if contexto else ""
    logger.error(f"Erro{contexto_str}: {str(erro)}", exc_info=True)


def log_configuration(logger: logging.Logger, config_dict: dict):
    """
    Log das configurações carregadas.
    
    Args:
        logger (logging.Logger): Logger a ser usado
        config_dict (dict): Dicionário com configurações
    """
    logger.info("📋 Configurações carregadas:")
    for key, value in config_dict.items():
        logger.info(f"   {key}: {value}")


def log_statistics(logger: logging.Logger, stats: dict):
    """
    Log de estatísticas.
    
    Args:
        logger (logging.Logger): Logger a ser usado
        stats (dict): Dicionário com estatísticas
    """
    logger.info("📊 Estatísticas:")
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")
