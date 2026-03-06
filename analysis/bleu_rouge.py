#!/usr/bin/env python3
"""
Módulo para cálculo de métricas BLEU e ROUGE
Centraliza todos os cálculos relacionados a essas métricas acadêmicas.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Adicionar o diretório pai ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import get_config

class BleuRougeCalculator:
    """Calculadora de métricas BLEU e ROUGE."""
    
    def __init__(self):
        self.config = get_config()
    
    def calcular_bleu_rouge_individual(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas BLEU e ROUGE para cada linha do DataFrame.
        
        Args:
            df: DataFrame com colunas 'pergunta', 'resposta_esperada', 'resposta_modelo'
            
        Returns:
            DataFrame com colunas adicionais de métricas BLEU e ROUGE
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from rouge_score import rouge_scorer
        except ImportError as e:
            print(f"❌ Erro ao importar dependências para BLEU/ROUGE: {e}")
            return df
        
        # Inicializar calculadoras
        smoothing = SmoothingFunction().method1
        rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Preparar dados
        df_result = df.copy()
        
        # Colunas para métricas
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        print("🔍 Calculando métricas BLEU e ROUGE...")
        
        for idx, row in df.iterrows():
            pergunta = str(row.get('prompt', ''))
            resposta_esperada = str(row.get('reference', ''))
            resposta_modelo = str(row.get('prediction', ''))
            
            # Verificar se é resposta inválida
            is_error_flag = bool(row.get('is_error', False)) if 'is_error' in df.columns else None
            resposta_invalida = is_error_flag if is_error_flag is not None else self._eh_resposta_invalida(resposta_modelo)
            if resposta_invalida:
                if idx < 5:  # Debug para as primeiras 5 linhas
                    print(f"⚠️ Linha {idx} marcada como inválida: {resposta_modelo[:50]}...")
                bleu_scores.append(0.0)
                rouge1_scores.append(0.0)
                rouge2_scores.append(0.0)
                rougeL_scores.append(0.0)
                continue
            
            # Para métricas de texto (BLEU/ROUGE), usar resposta completa
            # Não extrair A, B, C, D - isso é apenas para benchmarks de múltipla escolha
            resposta_final = resposta_modelo
            
            # Calcular BLEU
            try:
                reference = [resposta_esperada.split()]
                candidate = resposta_final.split()
                bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing)
                bleu_scores.append(bleu_score)
            except Exception as e:
                print(f"⚠️ Erro BLEU linha {idx}: {e}")
                bleu_scores.append(0.0)
            
            # Calcular ROUGE
            try:
                rouge_scores = rouge_scorer_instance.score(resposta_esperada, resposta_final)
                rouge1_score = rouge_scores['rouge1'].fmeasure
                rouge2_score = rouge_scores['rouge2'].fmeasure
                rougeL_score = rouge_scores['rougeL'].fmeasure
                
                # Debug para as primeiras 5 linhas
                if idx < 5:
                    print(f"✅ Linha {idx} ROUGE calculado - R1: {rouge1_score:.4f}, R2: {rouge2_score:.4f}, RL: {rougeL_score:.4f}")
                
                rouge1_scores.append(rouge1_score)
                rouge2_scores.append(rouge2_score)
                rougeL_scores.append(rougeL_score)
            except Exception as e:
                print(f"⚠️ Erro ROUGE linha {idx}: {e}")
                rouge1_scores.append(0.0)
                rouge2_scores.append(0.0)
                rougeL_scores.append(0.0)
        
        # Adicionar colunas ao DataFrame
        df_result['bleu_score'] = bleu_scores
        df_result['rouge1_score'] = rouge1_scores
        df_result['rouge2_score'] = rouge2_scores
        df_result['rougeL_score'] = rougeL_scores
        
        # Debug: verificar se ROUGE-2 foi salvo corretamente
        rouge2_unique = df_result['rouge2_score'].unique()
        print(f"✅ Métricas BLEU e ROUGE calculadas para {len(df)} respostas")
        print(f"🔍 ROUGE-2 valores únicos: {rouge2_unique[:10]}...")  # Mostrar apenas os primeiros 10
        print(f"🔍 ROUGE-2 max: {df_result['rouge2_score'].max():.4f}, min: {df_result['rouge2_score'].min():.4f}")
        
        return df_result
    
    def calcular_metricas_por_modelo(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calcula métricas BLEU e ROUGE agregadas por modelo.
        
        Args:
            df: DataFrame com métricas já calculadas
            
        Returns:
            Dicionário com métricas por modelo
        """
        if 'model' not in df.columns:
            print("❌ Coluna 'model' não encontrada no DataFrame")
            return {}
        
        metricas_por_modelo = {}
        
        for modelo in df['model'].unique():
            df_modelo = df[df['model'] == modelo]
            
            # Filtrar apenas respostas válidas
            if 'is_error' in df_modelo.columns:
                df_validas = df_modelo[~df_modelo['is_error']]
            else:
                df_validas = df_modelo[~df_modelo['prediction'].apply(self._eh_resposta_invalida)]
            
            if len(df_validas) == 0:
                metricas_por_modelo[modelo] = {
                    'bleu_medio': 0.0,
                    'rouge1_medio': 0.0,
                    'rouge2_medio': 0.0,
                    'rougeL_medio': 0.0,
                    'total_respostas': len(df_modelo),
                    'respostas_validas': 0,
                    'taxa_validas': 0.0
                }
                continue
            
            metricas_por_modelo[modelo] = {
                'bleu_medio': df_validas['bleu_score'].mean(),
                'rouge1_medio': df_validas['rouge1_score'].mean(),
                'rouge2_medio': df_validas['rouge2_score'].mean(),
                'rougeL_medio': df_validas['rougeL_score'].mean(),
                'bleu_std': df_validas['bleu_score'].std(),
                'rouge1_std': df_validas['rouge1_score'].std(),
                'rouge2_std': df_validas['rouge2_score'].std(),
                'rougeL_std': df_validas['rougeL_score'].std(),
                'total_respostas': len(df_modelo),
                'respostas_validas': len(df_validas),
                'taxa_validas': len(df_validas) / len(df_modelo)
            }
        
        return metricas_por_modelo
    
    def _eh_resposta_invalida(self, resposta: str) -> bool:
        """
        Verifica se uma resposta é inválida (erro de API, vazia, etc.).
        
        Args:
            resposta: Resposta do modelo
            
        Returns:
            True se a resposta é inválida
        """
        if not resposta or pd.isna(resposta):
            return True
        
        resposta_str = str(resposta).strip().lower()
        
        # Padrões de erro de API
        padroes_erro = [
            'erro',
            'error',
            'timeout',
            'rate limit',
            'quota excedida',
            'quota exceeded',
            'api key',
            'authentication',
            'connection',
            'network',
            'failed',
            'exception',
            'traceback',
            'null',
            'none',
            'undefined',
            'gemini_1_5_flash'  # Erro específico do Gemini
        ]
        
        return any(padrao in resposta_str for padrao in padroes_erro)
    
    
    def gerar_relatorio_bleu_rouge(self, metricas_por_modelo: Dict[str, Dict]) -> str:
        """
        Gera relatório em texto das métricas BLEU e ROUGE.
        
        Args:
            metricas_por_modelo: Dicionário com métricas por modelo
            
        Returns:
            String com relatório formatado
        """
        relatorio = []
        relatorio.append("## 📊 Métricas BLEU e ROUGE")
        relatorio.append("=" * 50)
        relatorio.append("")
        
        # Ordenar modelos por BLEU médio
        modelos_ordenados = sorted(
            metricas_por_modelo.items(),
            key=lambda x: x[1]['bleu_medio'],
            reverse=True
        )
        
        for modelo, metricas in modelos_ordenados:
            relatorio.append(f"### 🤖 {modelo}")
            relatorio.append(f"- **BLEU Score**: {metricas.get('bleu_medio', 0.0):.4f} ± {metricas.get('bleu_std', 0.0):.4f}")
            relatorio.append(f"- **ROUGE-1**: {metricas.get('rouge1_medio', 0.0):.4f} ± {metricas.get('rouge1_std', 0.0):.4f}")
            relatorio.append(f"- **ROUGE-2**: {metricas.get('rouge2_medio', 0.0):.4f} ± {metricas.get('rouge2_std', 0.0):.4f}")
            relatorio.append(f"- **ROUGE-L**: {metricas.get('rougeL_medio', 0.0):.4f} ± {metricas.get('rougeL_std', 0.0):.4f}")
            relatorio.append(f"- **Respostas Válidas**: {metricas['respostas_validas']}/{metricas['total_respostas']} ({metricas['taxa_validas']:.1%})")
            relatorio.append("")
        
        return "\n".join(relatorio)

def calcular_bleu_rouge_completo(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict], str]:
    """
    Função principal para calcular BLEU e ROUGE completos.
    
    Args:
        df: DataFrame com dados das respostas
        
    Returns:
        Tuple com (DataFrame com métricas, métricas por modelo, relatório)
    """
    calculator = BleuRougeCalculator()
    
    # Calcular métricas individuais
    df_com_metricas = calculator.calcular_bleu_rouge_individual(df)
    
    # Calcular métricas por modelo
    metricas_por_modelo = calculator.calcular_metricas_por_modelo(df_com_metricas)
    
    # Gerar relatório
    relatorio = calculator.gerar_relatorio_bleu_rouge(metricas_por_modelo)
    
    return df_com_metricas, metricas_por_modelo, relatorio
