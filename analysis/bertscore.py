#!/usr/bin/env python3
"""
Módulo para cálculo de métricas BERTScore
Centraliza todos os cálculos relacionados a essa métrica acadêmica.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Adicionar o diretório pai ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import get_config

class BertScoreCalculator:
    """Calculadora de métricas BERTScore."""
    
    def __init__(self):
        self.config = get_config()
    
    def calcular_bertscore_individual(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas BERTScore para cada linha do DataFrame.
        
        Args:
            df: DataFrame com colunas 'pergunta', 'resposta_esperada', 'resposta_modelo'
            
        Returns:
            DataFrame com colunas adicionais de métricas BERTScore
        """
        try:
            from bert_score import score as bert_score
        except ImportError as e:
            print(f"❌ Erro ao importar BERTScore: {e}")
            print("💡 Instale com: pip install bert-score")
            return df
        
        # Preparar dados
        df_result = df.copy()
        
        # Separar respostas válidas e inválidas
        respostas_validas = []
        indices_validos = []
        
        for idx, row in df.iterrows():
            resposta_modelo = str(row.get('prediction', ''))
            if not self._eh_resposta_invalida(resposta_modelo):
                respostas_validas.append(resposta_modelo)
                indices_validos.append(idx)
        
        if len(respostas_validas) == 0:
            print("⚠️ Nenhuma resposta válida encontrada para BERTScore")
            df_result['bertscore_precision'] = 0.0
            df_result['bertscore_recall'] = 0.0
            df_result['bertscore_f1'] = 0.0
            return df_result
        
        # Preparar referências
        referencias = []
        for idx in indices_validos:
            row = df.iloc[idx]
            referencias.append(str(row.get('reference', '')))
        
        print(f"🔍 Calculando BERTScore para {len(respostas_validas)} respostas válidas...")
        
        try:
            # Calcular BERTScore
            P, R, F1 = bert_score(
                respostas_validas,
                referencias,
                lang=self.config.BERT_SCORE_LANG,
                verbose=False
            )
            
            # Inicializar colunas com zeros
            df_result['bertscore_precision'] = 0.0
            df_result['bertscore_recall'] = 0.0
            df_result['bertscore_f1'] = 0.0
            
            # Preencher valores para respostas válidas
            for i, idx in enumerate(indices_validos):
                df_result.loc[idx, 'bertscore_precision'] = P[i].item()
                df_result.loc[idx, 'bertscore_recall'] = R[i].item()
                df_result.loc[idx, 'bertscore_f1'] = F1[i].item()
            
            print(f"✅ BERTScore calculado para {len(respostas_validas)} respostas")
            
        except Exception as e:
            print(f"❌ Erro ao calcular BERTScore: {e}")
            df_result['bertscore_precision'] = 0.0
            df_result['bertscore_recall'] = 0.0
            df_result['bertscore_f1'] = 0.0
        
        return df_result
    
    def calcular_metricas_por_modelo(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calcula métricas BERTScore agregadas por modelo.
        
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
            df_validas = df_modelo[~df_modelo['prediction'].apply(self._eh_resposta_invalida)]
            
            if len(df_validas) == 0:
                metricas_por_modelo[modelo] = {
                    'bertscore_precision_medio': 0.0,
                    'bertscore_recall_medio': 0.0,
                    'bertscore_f1_medio': 0.0,
                    'total_respostas': len(df_modelo),
                    'respostas_validas': 0,
                    'taxa_validas': 0.0
                }
                continue
            
            metricas_por_modelo[modelo] = {
                'bertscore_precision_medio': df_validas['bertscore_precision'].mean(),
                'bertscore_recall_medio': df_validas['bertscore_recall'].mean(),
                'bertscore_f1_medio': df_validas['bertscore_f1'].mean(),
                'bertscore_precision_std': df_validas['bertscore_precision'].std(),
                'bertscore_recall_std': df_validas['bertscore_recall'].std(),
                'bertscore_f1_std': df_validas['bertscore_f1'].std(),
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
            'api key',
            'authentication',
            'connection',
            'network',
            'failed',
            'exception',
            'traceback',
            'null',
            'none',
            'undefined'
        ]
        
        return any(padrao in resposta_str for padrao in padroes_erro)
    
    def gerar_relatorio_bertscore(self, metricas_por_modelo: Dict[str, Dict]) -> str:
        """
        Gera relatório em texto das métricas BERTScore.
        
        Args:
            metricas_por_modelo: Dicionário com métricas por modelo
            
        Returns:
            String com relatório formatado
        """
        relatorio = []
        relatorio.append("## 🧠 Métricas BERTScore")
        relatorio.append("=" * 50)
        relatorio.append("")
        
        # Ordenar modelos por F1 médio
        modelos_ordenados = sorted(
            metricas_por_modelo.items(),
            key=lambda x: x[1]['bertscore_f1_medio'],
            reverse=True
        )
        
        for modelo, metricas in modelos_ordenados:
            relatorio.append(f"### 🤖 {modelo}")
            relatorio.append(f"- **Precision**: {metricas['bertscore_precision_medio']:.4f} ± {metricas['bertscore_precision_std']:.4f}")
            relatorio.append(f"- **Recall**: {metricas['bertscore_recall_medio']:.4f} ± {metricas['bertscore_recall_std']:.4f}")
            relatorio.append(f"- **F1-Score**: {metricas['bertscore_f1_medio']:.4f} ± {metricas['bertscore_f1_std']:.4f}")
            relatorio.append(f"- **Respostas Válidas**: {metricas['respostas_validas']}/{metricas['total_respostas']} ({metricas['taxa_validas']:.1%})")
            relatorio.append("")
        
        return "\n".join(relatorio)

def calcular_bertscore_completo(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict], str]:
    """
    Função principal para calcular BERTScore completo.
    
    Args:
        df: DataFrame com dados das respostas
        
    Returns:
        Tuple com (DataFrame com métricas, métricas por modelo, relatório)
    """
    calculator = BertScoreCalculator()
    
    # Calcular métricas individuais
    df_com_metricas = calculator.calcular_bertscore_individual(df)
    
    # Calcular métricas por modelo
    metricas_por_modelo = calculator.calcular_metricas_por_modelo(df_com_metricas)
    
    # Gerar relatório
    relatorio = calculator.gerar_relatorio_bertscore(metricas_por_modelo)
    
    return df_com_metricas, metricas_por_modelo, relatorio
