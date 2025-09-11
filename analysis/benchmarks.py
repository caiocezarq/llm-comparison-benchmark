#!/usr/bin/env python3
"""
Classe base para benchmarks padronizados.
Define interface comum para todos os benchmarks (MMLU, HellaSwag, etc.).
"""

from typing import Dict, List, Any
import pandas as pd


class BaseBenchmark:
    """
    Classe base para implementação de benchmarks padronizados.
    """
    
    def __init__(self, name: str):
        """
        Inicializa o benchmark.
        
        Args:
            name: Nome do benchmark (ex: 'mmlu', 'hellaswag')
        """
        self.name = name
        self.prompts = []
        self.results = []
    
    def calculate_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """
        Calcula métricas específicas do benchmark.
        
        Args:
            predictions: Lista de predições do modelo
            references: Lista de respostas de referência
            
        Returns:
            Dicionário com métricas calculadas
        """
        raise NotImplementedError("Subclasses devem implementar calculate_metrics")
    
    def format_prompt(self, question_data: Dict[str, Any]) -> str:
        """
        Formata prompt para o benchmark específico.
        
        Args:
            question_data: Dados da questão do benchmark
            
        Returns:
            String formatada do prompt
        """
        raise NotImplementedError("Subclasses devem implementar format_prompt")
    
    def validate_prediction(self, prediction: str, reference: str) -> bool:
        """
        Valida se a predição está correta.
        
        Args:
            prediction: Predição do modelo
            reference: Resposta de referência
            
        Returns:
            True se a predição está correta, False caso contrário
        """
        if not prediction or not reference:
            return False
        
        # Normalizar strings para comparação
        pred_clean = prediction.strip().upper()
        ref_clean = reference.strip().upper()
        
        return pred_clean == ref_clean
    
    def calculate_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """
        Calcula accuracy geral do benchmark.
        
        Args:
            predictions: Lista de predições do modelo
            references: Lista de respostas de referência
            
        Returns:
            Accuracy como float entre 0 e 1
        """
        if not predictions or not references:
            return 0.0
        
        if len(predictions) != len(references):
            return 0.0
        
        correct = 0
        for pred, ref in zip(predictions, references):
            if self.validate_prediction(pred, ref):
                correct += 1
        
        return correct / len(predictions)
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o benchmark.
        
        Returns:
            Dicionário com informações do benchmark
        """
        return {
            "name": self.name,
            "description": self.get_description(),
            "metrics": self.get_available_metrics()
        }
    
    def get_description(self) -> str:
        """
        Retorna descrição do benchmark.
        
        Returns:
            String com descrição
        """
        return f"Benchmark {self.name}"
    
    def get_available_metrics(self) -> List[str]:
        """
        Retorna lista de métricas disponíveis para este benchmark.
        
        Returns:
            Lista de strings com nomes das métricas
        """
        return ["accuracy", "total_questions", "correct_answers"]
