#!/usr/bin/env python3
"""
Sistema de Relatórios Evidently AI
Gera relatórios HTML e análises de qualidade textual por modelo.
Baseado na documentação oficial: https://docs.evidentlyai.com/
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

try:
    from evidently import Report, Dataset, DataDefinition
    from evidently.metrics import *
    from evidently.presets import DataSummaryPreset, DataDriftPreset, TextEvals
    from evidently.generators import ColumnMetricGenerator
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("⚠️ Evidently AI não disponível. Instale com: pip install evidently")

class EvidentlyReporter:
    """Gerador de relatórios Evidently AI."""
    
    def __init__(self):
        self.evidently_available = EVIDENTLY_AVAILABLE
    
    def preparar_dados_evidently(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara dados para análise Evidently AI."""
        df_evidently = df.copy()
        
        # Adicionar colunas necessárias
        df_evidently['text_length'] = df_evidently['prediction'].astype(str).str.len()
        df_evidently['word_count'] = df_evidently['prediction'].astype(str).str.split().str.len()
        df_evidently['is_valid'] = ~df_evidently['prediction'].apply(self._eh_resposta_invalida)
        df_evidently['timestamp'] = pd.Timestamp.now()
        
        # Garantir que as colunas de texto sejam strings
        df_evidently['prediction'] = df_evidently['prediction'].astype(str)
        if 'reference' in df_evidently.columns:
            df_evidently['reference'] = df_evidently['reference'].astype(str)
        
        return df_evidently
    
    def _eh_resposta_invalida(self, resposta: str) -> bool:
        """Verifica se uma resposta é inválida."""
        if not resposta or pd.isna(resposta):
            return True
        
        resposta_str = str(resposta).strip().lower()
        padroes_erro = [
            'erro', 'error', 'timeout', 'rate limit', 'api key',
            'authentication', 'connection', 'network', 'failed',
            'exception', 'traceback', 'null', 'none', 'undefined'
        ]
        
        return any(padrao in resposta_str for padrao in padroes_erro)
    
    def gerar_relatorio_qualidade(self, df: pd.DataFrame, pasta_destino: str) -> Optional[str]:
        """Gera relatório de qualidade de dados usando a API moderna do Evidently AI."""
        if not self.evidently_available:
            return None
        
        try:
            # Criar Dataset com DataDefinition
            data_definition = DataDefinition()
            eval_data = Dataset.from_pandas(df, data_definition=data_definition)
            
            # Criar relatório usando presets e métricas específicas
            report = Report([
                DataSummaryPreset(),
                TextEvals(),
                # Métricas para colunas numéricas
                ColumnMetricGenerator(MinValue, column_types='num'),
                ColumnMetricGenerator(MaxValue, column_types='num'),
                ColumnMetricGenerator(MeanValue, column_types='num'),
                # Métricas para colunas categóricas
                ColumnMetricGenerator(MissingValueCount, column_types='cat'),
                # Métricas para colunas específicas
                ColumnMetricGenerator(MinValue, columns=['text_length', 'word_count']),
                ColumnMetricGenerator(MaxValue, columns=['text_length', 'word_count']),
            ])
            
            # Executar relatório
            my_eval = report.run(eval_data, None)
            
            # Salvar relatório
            arquivo_relatorio = os.path.join(pasta_destino, "evidently_qualidade.html")
            my_eval.save_html(arquivo_relatorio)
            
            return arquivo_relatorio
            
        except Exception as e:
            print(f"❌ Erro ao gerar relatório de qualidade: {e}")
            return None
    
    def gerar_relatorio_texto(self, df: pd.DataFrame, pasta_destino: str) -> Optional[str]:
        """Gera relatório de análise de texto usando a API moderna do Evidently AI."""
        if not self.evidently_available:
            return None
        
        try:
            # Criar Dataset com DataDefinition
            data_definition = DataDefinition()
            eval_data = Dataset.from_pandas(df, data_definition=data_definition)
            
            # Criar relatório focado em análise de texto
            report = Report([
                TextEvals(),
                # Métricas específicas para análise de texto
                ColumnMetricGenerator(MinValue, columns=['text_length', 'word_count']),
                ColumnMetricGenerator(MaxValue, columns=['text_length', 'word_count']),
                ColumnMetricGenerator(MeanValue, columns=['text_length', 'word_count']),
            ])
            
            # Executar relatório
            my_eval = report.run(eval_data, None)
            
            # Salvar relatório
            arquivo_relatorio = os.path.join(pasta_destino, "evidently_texto.html")
            my_eval.save_html(arquivo_relatorio)
            
            return arquivo_relatorio
            
        except Exception as e:
            print(f"❌ Erro ao gerar relatório de texto: {e}")
            return None
    
    def gerar_relatorios_evidently(self, df: pd.DataFrame, pasta_destino: str) -> Dict[str, str]:
        """Gera todos os relatórios Evidently AI."""
        print(f"📊 Gerando relatórios Evidently AI para {len(df)} registros...")
        
        # Preparar dados
        df_evidently = self.preparar_dados_evidently(df)
        
        # Criar pasta de destino
        os.makedirs(pasta_destino, exist_ok=True)
        
        relatorios = {}
        
        # Relatório de qualidade
        relatorio_qualidade = self.gerar_relatorio_qualidade(df_evidently, pasta_destino)
        if relatorio_qualidade:
            relatorios['qualidade'] = relatorio_qualidade
        
        # Relatório de texto
        relatorio_texto = self.gerar_relatorio_texto(df_evidently, pasta_destino)
        if relatorio_texto:
            relatorios['texto'] = relatorio_texto
        
        return relatorios
    
    def gerar_relatorio_consolidado(self, relatorios: Dict[str, str]) -> str:
        """Gera relatório consolidado em markdown."""
        relatorio = []
        relatorio.append("## 📊 Análise Evidently AI")
        relatorio.append("")
        
        if relatorios.get('qualidade'):
            relatorio.append(f"- **Relatório de Qualidade**: [evidently_qualidade.html]({relatorios['qualidade']})")
        
        if relatorios.get('texto'):
            relatorio.append(f"- **Relatório de Texto**: [evidently_texto.html]({relatorios['texto']})")
        
        relatorio.append("")
        relatorio.append("### 📈 Métricas Calculadas")
        relatorio.append("- **Qualidade de Dados**: Valores ausentes, distribuições, correlações")
        relatorio.append("- **Análise de Texto**: Comprimento, qualidade, descritores textuais")
        relatorio.append("- **Distribuições**: Análise estatística das respostas")
        relatorio.append("- **Validação**: Detecção de respostas inválidas")
        
        return "\n".join(relatorio)

def gerar_relatorios_evidently_completo(df: pd.DataFrame, pasta_destino: str) -> Tuple[Dict[str, str], str]:
    """Função principal para gerar relatórios Evidently AI completos."""
    reporter = EvidentlyReporter()
    
    # Gerar relatórios
    relatorios = reporter.gerar_relatorios_evidently(df, pasta_destino)
    
    # Gerar relatório consolidado
    relatorio_consolidado = reporter.gerar_relatorio_consolidado(relatorios)
    
    return relatorios, relatorio_consolidado

if __name__ == "__main__":
    # Teste
    print("🧪 Testando EvidentlyReporter...")
    
    # Criar dados de teste
    dados_teste = pd.DataFrame({
        'model': ['modelo1', 'modelo2'] * 10,
        'prediction': ['Esta é uma resposta de teste muito boa e detalhada.', 'Resposta curta.'] * 10,
        'reference': ['Esta é uma resposta de referência excelente.', 'Referência curta.'] * 10
    })
    
    # Gerar relatórios
    relatorios, relatorio = gerar_relatorios_evidently_completo(dados_teste, "teste_evidently")
    
    print("✅ Teste concluído!")
    print(f"Relatórios gerados: {list(relatorios.keys())}")
