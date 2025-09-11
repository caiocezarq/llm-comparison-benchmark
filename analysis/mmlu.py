#!/usr/bin/env python3
"""
Calculadora de métricas MMLU (Massive Multitask Language Understanding).
"""

from typing import Dict, List, Any

# Import compatível com execução direta e via import
try:
    from .benchmarks import BaseBenchmark
except ImportError:
    from benchmarks import BaseBenchmark


class MMLUBenchmark(BaseBenchmark):
    """
    Calculadora de métricas para o benchmark MMLU.
    """
    
    def __init__(self):
        super().__init__("mmlu")
        self.subjects = []
    
    def calculate_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """
        Calcula métricas MMLU.
        
        Args:
            predictions: Lista de predições do modelo
            references: Lista de respostas de referência
            
        Returns:
            Dicionário com métricas MMLU
        """
        if not predictions or not references:
            return {
                "accuracy": 0.0,
                "total_questions": 0,
                "correct_answers": 0,
                "subjects": {}
            }
        
        # Calcular accuracy geral
        accuracy = self.calculate_accuracy(predictions, references)
        correct_answers = sum(1 for p, r in zip(predictions, references) 
                             if self.validate_prediction(p, r))
        
        # Calcular accuracy por subject (se disponível)
        subjects_accuracy = {}
        if hasattr(self, 'subjects') and self.subjects:
            for subject in self.subjects:
                subject_predictions = [p for p in predictions if p.get('subject') == subject]
                subject_references = [r for r in references if r.get('subject') == subject]
                if subject_predictions and subject_references:
                    subjects_accuracy[subject] = self.calculate_accuracy(
                        subject_predictions, subject_references
                    )
        
        return {
            "accuracy": accuracy,
            "total_questions": len(predictions),
            "correct_answers": correct_answers,
            "subjects": subjects_accuracy
        }
    
    def format_prompt(self, question_data: Dict[str, Any]) -> str:
        """
        Formata prompt para MMLU.
        
        Args:
            question_data: Dados da questão MMLU
            
        Returns:
            String formatada do prompt
        """
        choices = "\n".join([f"{choice}" for choice in question_data['choices']])
        return f"Question: {question_data['question']}\n\nChoices:\n{choices}\n\nInstructions: Choose the correct answer and respond with ONLY the letter (A, B, C, or D). Do not provide any explanation or additional text.\n\nAnswer:"
    
    def validate_prediction(self, prediction: str, reference: str) -> bool:
        """
        Valida se a predição MMLU está correta.
        
        Args:
            prediction: Predição do modelo
            reference: Resposta de referência (A, B, C, D)
            
        Returns:
            True se a predição está correta, False caso contrário
        """
        if not prediction or not reference:
            return False
        
        # Normalizar strings para comparação
        pred_clean = prediction.strip().upper()
        ref_clean = reference.strip().upper()
        
        # Para MMLU, verificar se a predição contém a resposta correta
        # ou se é exatamente igual
        if pred_clean == ref_clean:
            return True
        
        # Verificar se a predição contém a letra da resposta correta
        # mas apenas se for uma resposta de múltipla escolha válida
        if ref_clean in ['A', 'B', 'C', 'D']:
            # Padrões mais específicos para capturar respostas corretas
            patterns = [
                f"{ref_clean})",  # A), B), C), D)
                f"ANSWER IS {ref_clean}",  # ANSWER IS A
                f"RESPOSTA É {ref_clean}",  # RESPOSTA É A
                f"THE ANSWER IS {ref_clean}",  # THE ANSWER IS A
                f"ANSWER: {ref_clean}",  # ANSWER: A
                f"RESPOSTA: {ref_clean}",  # RESPOSTA: A
                f"THE ANSWER: {ref_clean}",  # THE ANSWER: A
                f"CHOICE {ref_clean}",  # CHOICE A
                f"OPÇÃO {ref_clean}",  # OPÇÃO A
                f"OPTION {ref_clean}",  # OPTION A
            ]
            
            for pattern in patterns:
                if pattern in pred_clean:
                    return True
            
            # Verificar se a predição começa com a letra correta seguida de parênteses
            if pred_clean.startswith(f"{ref_clean})"):
                return True
            
            # Verificar se a predição é apenas a letra (caso mais simples)
            if pred_clean == ref_clean:
                return True
        
        return False
    
    def get_description(self) -> str:
        """
        Retorna descrição do benchmark MMLU.
        
        Returns:
            String com descrição
        """
        return "Massive Multitask Language Understanding - Avaliação de conhecimento em múltiplas disciplinas"
    
    def get_available_metrics(self) -> List[str]:
        """
        Retorna lista de métricas disponíveis para MMLU.
        
        Returns:
            Lista de strings com nomes das métricas
        """
        return ["accuracy", "total_questions", "correct_answers", "subjects"]
