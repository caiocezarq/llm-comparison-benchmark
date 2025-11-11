#!/usr/bin/env python3
"""
Sistema de Análise Consolidada de Modelos LLM
Varre pasta results, agrupa por modelo e gera análises completas.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

# Adicionar o diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import get_config

# Imports para benchmarks - compatibilidade com chamada direta e via main
try:
    # Tentar import relativo (quando chamado via main.py)
    from .benchmarks import BaseBenchmark
    from .mmlu import MMLUBenchmark
    from .hellaswag import HellaSwagBenchmark
except ImportError:
    # Fallback para import absoluto (quando executado diretamente)
    from benchmarks import BaseBenchmark
    from mmlu import MMLUBenchmark
    from hellaswag import HellaSwagBenchmark

class AnalysisSystem:
    """Sistema principal de análise consolidada."""
    
    def __init__(self):
        self.config = get_config()
        self.pasta_analysis = "analysis"
        self.pasta_resultados = self.config.PASTA_RESULTADOS
        self.prefixo_execucao = self.config.PREFIXO_EXECUCAO
        
        # Inicializar benchmarks
        try:
            self.benchmarks = {
                'mmlu': MMLUBenchmark(),
                'hellaswag': HellaSwagBenchmark()
            }
        except Exception as e:
            print(f"⚠️ Erro ao inicializar benchmarks: {e}")
            self.benchmarks = {}
    
    def encontrar_execucoes(self) -> List[str]:
        """Encontra todas as execuções disponíveis na pasta de resultados."""
        if not os.path.exists(self.pasta_resultados):
            print(f"❌ Pasta {self.pasta_resultados} não encontrada")
            return []
        
        execucoes = []
        for item in os.listdir(self.pasta_resultados):
            if item.startswith(self.prefixo_execucao):
                caminho_execucao = os.path.join(self.pasta_resultados, item)
                if os.path.isdir(caminho_execucao):
                    execucoes.append(caminho_execucao)
        
        execucoes.sort()
        return execucoes
    
    def carregar_dados_execucao(self, caminho_execucao: str) -> Optional[pd.DataFrame]:
        """Carrega dados de uma execução específica."""
        arquivo_csv = os.path.join(caminho_execucao, "resultados_todos.csv")
        
        if not os.path.exists(arquivo_csv):
            print(f"❌ Arquivo {arquivo_csv} não encontrado")
            return None
        
        try:
            df = pd.read_csv(arquivo_csv, encoding=self.config.ENCODING_CSV)
            print(f"✅ Dados carregados: {len(df)} registros de {caminho_execucao}")
            return df
        except Exception as e:
            print(f"❌ Erro ao carregar {arquivo_csv}: {e}")
            return None

    def consolidar_dados_por_modelo(self, execucoes: List[str]) -> Dict[str, pd.DataFrame]:
        """Consolida dados agrupando por modelo."""
        print("🔄 Consolidando dados por modelo...")
        
        dados_por_modelo = {}
        
        for execucao in execucoes:
            nome_execucao = os.path.basename(execucao)
            print(f"📁 Processando {nome_execucao}...")
            
            # Carregar dados da execução
            df = self.carregar_dados_execucao(execucao)
            if df is None:
                continue
            
            # Adicionar coluna de execução
            df['execucao'] = nome_execucao
            
            # Agrupar por modelo
            for modelo in df['model'].unique():
                df_modelo = df[df['model'] == modelo].copy()
                
                if modelo not in dados_por_modelo:
                    dados_por_modelo[modelo] = []
                
                dados_por_modelo[modelo].append(df_modelo)
        
        # Concatenar dados de cada modelo
        dados_consolidados = {}
        for modelo, lista_dfs in dados_por_modelo.items():
            if lista_dfs:
                dados_consolidados[modelo] = pd.concat(lista_dfs, ignore_index=True)
                print(f"✅ {modelo}: {len(dados_consolidados[modelo])} registros consolidados")
        
        return dados_consolidados
    
    def calcular_metricas_academicas(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, str]:
        """Calcula todas as métricas acadêmicas (BLEU, ROUGE, BERTScore)."""
        print("📊 Calculando métricas acadêmicas...")
        
        # BLEU e ROUGE
        try:
            # Tentar import relativo (quando chamado via main.py)
            from .bleu_rouge import calcular_bleu_rouge_completo
        except ImportError:
            # Fallback para import absoluto (quando executado diretamente)
            from bleu_rouge import calcular_bleu_rouge_completo
        
        try:
            df_bleu_rouge, metricas_bleu_rouge, relatorio_bleu_rouge = calcular_bleu_rouge_completo(df)
        except Exception as e:
            print(f"❌ Erro ao calcular BLEU/ROUGE: {e}")
            return df, {}, "Erro ao calcular BLEU/ROUGE"
        
        # BERTScore
        try:
            # Tentar import relativo (quando chamado via main.py)
            from .bertscore import calcular_bertscore_completo
        except ImportError:
            # Fallback para import absoluto (quando executado diretamente)
            from bertscore import calcular_bertscore_completo
        
        try:
            df_bertscore, metricas_bertscore, relatorio_bertscore = calcular_bertscore_completo(df_bleu_rouge)
        except Exception as e:
            print(f"❌ Erro ao calcular BERTScore: {e}")
            return df_bleu_rouge, metricas_bleu_rouge, relatorio_bleu_rouge
        
        # Calcular métricas agregadas por modelo
        metricas_agregadas_dict = self._calcular_metricas_agregadas(df_bertscore)
        
        # Extrair métricas do modelo atual (assumindo que há apenas um modelo no DataFrame)
        if metricas_agregadas_dict:
            modelo_atual = df_bertscore['model'].iloc[0]
            metricas_agregadas = metricas_agregadas_dict.get(modelo_atual, {})
        else:
            metricas_agregadas = {}
        
        # Combinar relatórios
        relatorio_completo = f"{relatorio_bleu_rouge}\n\n{relatorio_bertscore}"
        
        return df_bertscore, metricas_agregadas, relatorio_completo
    
    def calcular_metricas_evidently(self, df: pd.DataFrame) -> Dict:
        """Calcula métricas do Evidently AI para cada modelo."""
        print("📈 Calculando métricas Evidently AI...")
        
        metricas_evidently = {}
        
        # Preparar dados
        df_evidently = df.copy()
        
        # Usar campos de comprimento se disponíveis (nova pipeline), senão calcular
        if 'response_length' in df_evidently.columns:
            df_evidently['text_length'] = df_evidently['response_length']
        else:
            df_evidently['text_length'] = df_evidently['prediction'].astype(str).str.len()
        
        if 'word_count' in df_evidently.columns:
            # Manter word_count se já existe
            pass
        else:
            df_evidently['word_count'] = df_evidently['prediction'].astype(str).str.split().str.len()
        
        # Usar campo is_error se disponível, senão calcular is_valid
        if self._usar_campo_is_error(df):
            df_evidently['is_valid'] = ~df_evidently['is_error']
        else:
            df_evidently['is_valid'] = ~df_evidently['prediction'].apply(self._eh_resposta_invalida)
        
        # Calcular métricas
        total_respostas = len(df_evidently)
        respostas_validas = len(df_evidently[df_evidently['is_valid']])
        taxa_validas = respostas_validas / total_respostas if total_respostas > 0 else 0
        
        df_validas = df_evidently[df_evidently['is_valid']]
        
        metricas_evidently = {
            'total_respostas': total_respostas,
            'respostas_validas': respostas_validas,
            'taxa_validas': taxa_validas,
            'comprimento_medio': df_validas['text_length'].mean() if len(df_validas) > 0 else 0,
            'comprimento_std': df_validas['text_length'].std() if len(df_validas) > 0 else 0,
            'palavras_medias': df_validas['word_count'].mean() if len(df_validas) > 0 else 0,
            'palavras_std': df_validas['word_count'].std() if len(df_validas) > 0 else 0
        }
        
        return metricas_evidently
    
    def _eh_modelo_problematico(self, df: pd.DataFrame, modelo: str) -> bool:
        """
        Verifica se um modelo tem alta taxa de erro e deve ser excluído da análise principal.
        
        Args:
            df: DataFrame com dados
            modelo: Nome do modelo
            
        Returns:
            True se o modelo deve ser excluído
        """
        df_modelo = df[df['model'] == modelo]
        
        if len(df_modelo) == 0:
            return True
        
        # Verificar taxa de erro
        if 'is_error' in df.columns:
            taxa_erro = df_modelo['is_error'].mean()
        else:
            # Estimar taxa de erro baseado em respostas inválidas
            respostas_invalidas = df_modelo['prediction'].apply(self._eh_resposta_invalida)
            taxa_erro = respostas_invalidas.mean()
        
        # Excluir modelos com taxa de erro > 40%
        return taxa_erro > 0.4
    
    def _calcular_metricas_agregadas(self, df: pd.DataFrame) -> Dict:
        """Calcula métricas agregadas por modelo."""
        if 'model' not in df.columns:
            return {}

        metricas_agregadas = {}

        for modelo in df['model'].unique():
            # Filtrar modelos com alta taxa de erro (ex: Gemini 1.5 Flash)
            if self._eh_modelo_problematico(df, modelo):
                print(f"⚠️ Modelo {modelo} tem alta taxa de erro - excluindo da análise principal")
                continue
            df_modelo = df[df['model'] == modelo]
            
            # Filtrar apenas respostas válidas - usar campo is_error se disponível
            if self._usar_campo_is_error(df):
                df_validas = df_modelo[~df_modelo['is_error']]
            else:
                df_validas = df_modelo[~df_modelo['prediction'].apply(self._eh_resposta_invalida)]
            
            if len(df_validas) == 0:
                metricas_agregadas[modelo] = {
                    'bleu_medio': 0.0,
                    'rouge1_medio': 0.0,
                    'rouge2_medio': 0.0,
                    'rougeL_medio': 0.0,
                    'bertscore_f1_medio': 0.0,
                    'total_respostas': len(df_modelo),
                    'respostas_validas': 0,
                    'taxa_validas': 0.0
                }
                continue
            
            metricas_agregadas[modelo] = {
                'bleu_medio': df_validas['bleu_score'].mean(),
                'rouge1_medio': df_validas['rouge1_score'].mean(),
                'rouge2_medio': df_validas['rouge2_score'].mean(),
                'rougeL_medio': df_validas['rougeL_score'].mean(),
                'bertscore_f1_medio': df_validas['bertscore_f1'].mean(),
                'bleu_std': df_validas['bleu_score'].std(),
                'rouge1_std': df_validas['rouge1_score'].std(),
                'rouge2_std': df_validas['rouge2_score'].std(),
                'rougeL_std': df_validas['rougeL_score'].std(),
                'bertscore_f1_std': df_validas['bertscore_f1'].std(),
                'total_respostas': len(df_modelo),
                'respostas_validas': len(df_validas),
                'taxa_validas': len(df_validas) / len(df_modelo)
            }
            
        return metricas_agregadas
    
    def _analisar_html_evidently(self, pasta_evidently: str) -> str:
        """Analisa os relatórios HTML do Evidently AI e extrai informações relevantes."""
        try:
            import re
            from bs4 import BeautifulSoup
            
            relatorio_html = ""
            
            # Analisar relatório de qualidade
            arquivo_qualidade = os.path.join(pasta_evidently, "evidently_qualidade.html")
            if os.path.exists(arquivo_qualidade):
                with open(arquivo_qualidade, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    relatorio_html += "### 📊 Análise de Qualidade dos Dados\n"
                    relatorio_html += "**Métricas extraídas do relatório Evidently AI:**\n\n"
                    
                    # Extrair métricas mais específicas
                    metricas_qualidade = self._extrair_metricas_qualidade(soup)
                    if metricas_qualidade:
                        for metrica, valor in metricas_qualidade.items():
                            relatorio_html += f"- **{metrica}**: {valor}\n"
                    else:
                        relatorio_html += "- **Status**: Relatório de qualidade gerado com sucesso\n"
                        relatorio_html += "- **Conteúdo**: Análise de distribuições, valores ausentes e correlações\n"
                    
                    relatorio_html += "\n"
            
            # Analisar relatório de texto
            arquivo_texto = os.path.join(pasta_evidently, "evidently_texto.html")
            if os.path.exists(arquivo_texto):
                with open(arquivo_texto, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    relatorio_html += "### 📝 Análise de Texto\n"
                    relatorio_html += "**Métricas de análise textual:**\n\n"
                    
                    # Extrair métricas de texto mais específicas
                    metricas_texto = self._extrair_metricas_texto(soup)
                    if metricas_texto:
                        for metrica, valor in metricas_texto.items():
                            relatorio_html += f"- **{metrica}**: {valor}\n"
                    else:
                        relatorio_html += "- **Status**: Relatório de texto gerado com sucesso\n"
                        relatorio_html += "- **Conteúdo**: Análise de comprimento, qualidade e descritores textuais\n"
                    
                    relatorio_html += "\n"
            
            if not relatorio_html:
                relatorio_html = "### 📊 Análise Evidently AI\n**Relatórios HTML gerados com sucesso.**\n\n"
            
            return relatorio_html
            
        except Exception as e:
            return f"### 📊 Análise Evidently AI\n**Erro ao analisar HTML**: {str(e)}\n\n"
    
    def _extrair_metricas_qualidade(self, soup) -> dict:
        """Extrai métricas específicas do relatório de qualidade."""
        import re
        metricas = {}
        
        try:
            # Procurar por tabelas com métricas
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        label = cells[0].get_text().strip()
                        value = cells[1].get_text().strip()
                        
                        # Filtrar métricas relevantes
                        if any(keyword in label.lower() for keyword in ['missing', 'null', 'count', 'mean', 'std', 'min', 'max']):
                            if re.match(r'^\d+\.?\d*$', value) or '%' in value:
                                metricas[label] = value
            
            # Procurar por elementos com classes específicas
            for element in soup.find_all(['span', 'div'], class_=re.compile(r'value|metric|stat')):
                text = element.get_text().strip()
                if re.match(r'^\d+\.?\d*$', text) and len(text) > 0:
                    # Tentar encontrar o label associado
                    parent = element.parent
                    if parent:
                        label_elem = parent.find(['span', 'div', 'td'], class_=re.compile(r'label|name|title'))
                        if label_elem:
                            label = label_elem.get_text().strip()
                            metricas[label] = text
            
        except Exception as e:
            print(f"⚠️ Erro ao extrair métricas de qualidade: {e}")
        
        return metricas
    
    def _extrair_metricas_texto(self, soup) -> dict:
        """Extrai métricas específicas do relatório de texto."""
        import re
        metricas = {}
        
        try:
            # Procurar por métricas de texto específicas
            text_keywords = ['length', 'word', 'character', 'sentence', 'paragraph', 'quality', 'readability']
            
            for element in soup.find_all(['span', 'div', 'td']):
                text = element.get_text().strip()
                if any(keyword in text.lower() for keyword in text_keywords):
                    # Verificar se é um valor numérico
                    if re.match(r'^\d+\.?\d*$', text) or '%' in text:
                        # Tentar encontrar o label
                        parent = element.parent
                        if parent:
                            label_elem = parent.find(['span', 'div', 'td'], class_=re.compile(r'label|name|title'))
                            if label_elem:
                                label = label_elem.get_text().strip()
                                metricas[label] = text
            
            # Procurar por tabelas com métricas de texto
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        label = cells[0].get_text().strip()
                        value = cells[1].get_text().strip()
                        
                        if any(keyword in label.lower() for keyword in text_keywords):
                            if re.match(r'^\d+\.?\d*$', value) or '%' in value:
                                metricas[label] = value
            
        except Exception as e:
            print(f"⚠️ Erro ao extrair métricas de texto: {e}")
        
        return metricas
    
    def _gerar_metricas_detalhadas_evidently(self, df: pd.DataFrame) -> str:
        """Gera métricas detalhadas do Evidently AI baseadas nos dados."""
        try:
            relatorio = []
            relatorio.append("### 📊 Métricas Detalhadas Evidently AI")
            relatorio.append("**Análise baseada nos dados processados:**\n")

            # Preparar dados
            df_evidently = df.copy()
            
            # Usar campos de comprimento se disponíveis (nova pipeline), senão calcular
            if 'response_length' in df_evidently.columns:
                df_evidently['text_length'] = df_evidently['response_length']
            else:
                df_evidently['text_length'] = df_evidently['prediction'].astype(str).str.len()
            
            if 'word_count' in df_evidently.columns:
                # Manter word_count se já existe
                pass
            else:
                df_evidently['word_count'] = df_evidently['prediction'].astype(str).str.split().str.len()
            
            # Usar campo is_error se disponível, senão calcular is_valid
            if self._usar_campo_is_error(df):
                df_evidently['is_valid'] = ~df_evidently['is_error']
            else:
                df_evidently['is_valid'] = ~df_evidently['prediction'].apply(self._eh_resposta_invalida)
            
            # Filtrar apenas respostas válidas
            df_validas = df_evidently[df_evidently['is_valid']]
            
            if len(df_validas) == 0:
                relatorio.append("- **Status**: Nenhuma resposta válida encontrada")
                return "\n".join(relatorio)
            
            # Métricas de qualidade de dados
            relatorio.append("#### 🔍 Qualidade de Dados")
            relatorio.append(f"- **Total de Registros**: {len(df_evidently)}")
            relatorio.append(f"- **Respostas Válidas**: {len(df_validas)}")
            relatorio.append(f"- **Taxa de Validade**: {len(df_validas)/len(df_evidently):.1%}")
            relatorio.append(f"- **Respostas Inválidas**: {len(df_evidently) - len(df_validas)}")
            relatorio.append("")
            
            # Métricas de texto
            relatorio.append("#### 📝 Análise de Texto")
            relatorio.append(f"- **Comprimento Médio**: {df_validas['text_length'].mean():.1f} caracteres")
            relatorio.append(f"- **Comprimento Mínimo**: {df_validas['text_length'].min():.0f} caracteres")
            relatorio.append(f"- **Comprimento Máximo**: {df_validas['text_length'].max():.0f} caracteres")
            relatorio.append(f"- **Desvio Padrão**: {df_validas['text_length'].std():.1f} caracteres")
            relatorio.append("")
            
            relatorio.append(f"- **Palavras Médias**: {df_validas['word_count'].mean():.1f}")
            relatorio.append(f"- **Palavras Mínimas**: {df_validas['word_count'].min():.0f}")
            relatorio.append(f"- **Palavras Máximas**: {df_validas['word_count'].max():.0f}")
            relatorio.append(f"- **Desvio Padrão Palavras**: {df_validas['word_count'].std():.1f}")
            relatorio.append("")
            
            # Análise de distribuição
            relatorio.append("#### 📊 Distribuição de Comprimento")
            q25 = df_validas['text_length'].quantile(0.25)
            q50 = df_validas['text_length'].quantile(0.50)
            q75 = df_validas['text_length'].quantile(0.75)
            
            relatorio.append(f"- **Q1 (25%)**: {q25:.1f} caracteres")
            relatorio.append(f"- **Mediana (50%)**: {q50:.1f} caracteres")
            relatorio.append(f"- **Q3 (75%)**: {q75:.1f} caracteres")
            relatorio.append(f"- **Amplitude Interquartil**: {q75 - q25:.1f} caracteres")
            relatorio.append("")
            
            # Análise de consistência
            cv_comprimento = (df_validas['text_length'].std() / df_validas['text_length'].mean()) * 100
            cv_palavras = (df_validas['word_count'].std() / df_validas['word_count'].mean()) * 100
            
            relatorio.append("#### ⚖️ Consistência")
            relatorio.append(f"- **Coeficiente de Variação (Comprimento)**: {cv_comprimento:.1f}%")
            relatorio.append(f"- **Coeficiente de Variação (Palavras)**: {cv_palavras:.1f}%")
            
            if cv_comprimento < 20:
                consistencia = "✅ Muito consistente"
            elif cv_comprimento < 40:
                consistencia = "⚠️ Moderadamente consistente"
            else:
                consistencia = "❌ Pouco consistente"
            
            relatorio.append(f"- **Avaliação de Consistência**: {consistencia}")
            relatorio.append("")
            
            # Análise de outliers
            relatorio.append("#### 🎯 Análise de Outliers")
            q1 = df_validas['text_length'].quantile(0.25)
            q3 = df_validas['text_length'].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr
            
            outliers = df_validas[(df_validas['text_length'] < limite_inferior) | 
                                 (df_validas['text_length'] > limite_superior)]
            
            relatorio.append(f"- **Outliers Detectados**: {len(outliers)}")
            relatorio.append(f"- **Limite Inferior**: {limite_inferior:.1f} caracteres")
            relatorio.append(f"- **Limite Superior**: {limite_superior:.1f} caracteres")
            relatorio.append("")
            
            # Resumo estatístico
            relatorio.append("#### 📈 Resumo Estatístico")
            relatorio.append(f"- **Média Aritmética**: {df_validas['text_length'].mean():.1f}")
            relatorio.append(f"- **Mediana**: {df_validas['text_length'].median():.1f}")
            relatorio.append(f"- **Moda**: {df_validas['text_length'].mode().iloc[0] if len(df_validas['text_length'].mode()) > 0 else 'N/A'}")
            relatorio.append(f"- **Variância**: {df_validas['text_length'].var():.1f}")
            relatorio.append(f"- **Assimetria**: {df_validas['text_length'].skew():.3f}")
            relatorio.append(f"- **Curtose**: {df_validas['text_length'].kurtosis():.3f}")
            
            return "\n".join(relatorio)
            
        except Exception as e:
            return f"### 📊 Métricas Detalhadas Evidently AI\n**Erro ao gerar métricas**: {str(e)}\n"
    
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
    
    def calcular_metricas_benchmarks(self, df: pd.DataFrame) -> Dict:
        """
        Calcula métricas de benchmarks (MMLU, HellaSwag) para cada modelo.
        
        Args:
            df: DataFrame com resultados dos modelos
            
        Returns:
            Dicionário com métricas de benchmarks por modelo
        """
        print("🏆 Calculando métricas de benchmarks...")
        
        metricas_benchmarks = {}
        
        # Verificar se há benchmarks disponíveis
        if not self.benchmarks:
            print("⚠️ Nenhum benchmark disponível")
            return {}
        
        # Verificar se há coluna benchmark no DataFrame
        if 'benchmark' not in df.columns:
            print("⚠️ Coluna 'benchmark' não encontrada no DataFrame")
            return {}
        
        # Filtrar apenas linhas com benchmarks
        df_benchmarks = df[df['benchmark'].notna()].copy()
        
        if df_benchmarks.empty:
            print("⚠️ Nenhum dado de benchmark encontrado")
            return {}
        
        # Agrupar por modelo e benchmark
        for model in df_benchmarks['model'].unique():
            metricas_benchmarks[model] = {}
            
            for benchmark_name, benchmark_calc in self.benchmarks.items():
                try:
                    # Filtrar dados do benchmark específico
                    df_benchmark = df_benchmarks[
                        (df_benchmarks['model'] == model) & 
                        (df_benchmarks['benchmark'] == benchmark_name)
                    ]
                    
                    if not df_benchmark.empty:
                        predictions = df_benchmark['prediction'].tolist()
                        references = df_benchmark['reference'].tolist()
                        
                        # Calcular métricas do benchmark
                        metrics = benchmark_calc.calculate_metrics(predictions, references)
                        metricas_benchmarks[model][benchmark_name] = metrics
                    else:
                        metricas_benchmarks[model][benchmark_name] = {
                            "accuracy": 0.0,
                            "total_questions": 0,
                            "correct_answers": 0
                        }
                except Exception as e:
                    print(f"⚠️ Erro ao calcular métricas do benchmark {benchmark_name} para {model}: {e}")
                    metricas_benchmarks[model][benchmark_name] = {
                        "accuracy": 0.0,
                        "total_questions": 0,
                        "correct_answers": 0
                    }
        
        return metricas_benchmarks
    
    def _usar_campo_is_error(self, df: pd.DataFrame) -> bool:
        """Verifica se o DataFrame tem o campo is_error (nova versão da pipeline)."""
        return 'is_error' in df.columns
    
    def gerar_relatorio_por_modelo(self, modelo: str, df: pd.DataFrame, 
                                 metricas_academicas: Dict, metricas_evidently: Dict,
                                 metricas_benchmarks: Dict, pasta_destino: str) -> str:
        """Gera relatório individual por modelo."""
        relatorio = []
        relatorio.append(f"# 🤖 Análise do Modelo: {modelo}")
        relatorio.append("")
        relatorio.append(f"**Data da Análise**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        relatorio.append(f"**Total de Respostas**: {len(df)}")
        relatorio.append(f"**Execuções**: {', '.join(df['execucao'].unique())}")
        relatorio.append("")
        
        # Métricas acadêmicas
        relatorio.append("## 📊 Métricas Acadêmicas")
        relatorio.append("")
        relatorio.append(f"- **BLEU Score**: {metricas_academicas.get('bleu_medio', 0):.4f}")
        relatorio.append(f"- **ROUGE-1**: {metricas_academicas.get('rouge1_medio', 0):.4f}")
        relatorio.append(f"- **ROUGE-2**: {metricas_academicas.get('rouge2_medio', 0):.4f}")
        relatorio.append(f"- **ROUGE-L**: {metricas_academicas.get('rougeL_medio', 0):.4f}")
        relatorio.append(f"- **BERTScore**: {metricas_academicas.get('bertscore_f1_medio', 0):.4f}")
        relatorio.append("")
        
        # Métricas de Benchmarks
        if metricas_benchmarks and modelo in metricas_benchmarks:
            relatorio.append("## 🏆 Métricas de Benchmarks")
            relatorio.append("")
            
            # MMLU
            if 'mmlu' in metricas_benchmarks[modelo]:
                mmlu_data = metricas_benchmarks[modelo]['mmlu']
                relatorio.append("### MMLU (Massive Multitask Language Understanding)")
                relatorio.append(f"- **Accuracy**: {mmlu_data.get('accuracy', 0):.4f}")
                relatorio.append(f"- **Total de Questões**: {mmlu_data.get('total_questions', 0)}")
                relatorio.append(f"- **Respostas Corretas**: {mmlu_data.get('correct_answers', 0)}")
                
                # Subjects específicos se disponível
                if 'subjects' in mmlu_data and mmlu_data['subjects']:
                    relatorio.append("- **Accuracy por Subject**:")
                    for subject, accuracy in mmlu_data['subjects'].items():
                        relatorio.append(f"  - {subject}: {accuracy:.4f}")
                relatorio.append("")
            
            # HellaSwag
            if 'hellaswag' in metricas_benchmarks[modelo]:
                hellaswag_data = metricas_benchmarks[modelo]['hellaswag']
                relatorio.append("### HellaSwag (Commonsense Reasoning)")
                relatorio.append(f"- **Accuracy**: {hellaswag_data.get('accuracy', 0):.4f}")
                relatorio.append(f"- **Total de Questões**: {hellaswag_data.get('total_questions', 0)}")
                relatorio.append(f"- **Respostas Corretas**: {hellaswag_data.get('correct_answers', 0)}")
                relatorio.append("")
        
        # Métricas Evidently AI
        relatorio.append("## 📈 Métricas Evidently AI")
        relatorio.append("")
        relatorio.append(f"- **Respostas Válidas**: {metricas_evidently.get('respostas_validas', 0)}")
        relatorio.append(f"- **Taxa de Validade**: {metricas_evidently.get('taxa_validas', 0):.1%}")
        relatorio.append(f"- **Comprimento Médio**: {metricas_evidently.get('comprimento_medio', 0):.1f} ± {metricas_evidently.get('comprimento_std', 0):.1f} caracteres")
        relatorio.append(f"- **Palavras Médias**: {metricas_evidently.get('palavras_medias', 0):.1f} ± {metricas_evidently.get('palavras_std', 0):.1f}")
        relatorio.append("")
        
        # Análise de consistência
        comprimento_std = metricas_evidently.get('comprimento_std', 0)
        comprimento_medio = metricas_evidently.get('comprimento_medio', 0)
        
        if comprimento_medio > 0:
            cv_comprimento = (comprimento_std / comprimento_medio) * 100
            if cv_comprimento < 20:
                consistencia = "✅ Muito consistente"
            elif cv_comprimento < 40:
                consistencia = "⚠️ Moderadamente consistente"
            else:
                consistencia = "❌ Pouco consistente"
            
            relatorio.append(f"- **Consistência de Comprimento**: {consistencia} (CV: {cv_comprimento:.1f}%)")
            relatorio.append("")
        
        # Avaliação geral
        taxa_validas = metricas_evidently.get('taxa_validas', 0)
        if taxa_validas > 0.9:
            status_geral = "✅ Excelente"
        elif taxa_validas > 0.7:
            status_geral = "⚠️ Bom"
        elif taxa_validas > 0.5:
            status_geral = "⚠️ Regular"
        else:
            status_geral = "❌ Ruim"
        
        relatorio.append(f"**Avaliação Geral**: {status_geral}")
        relatorio.append("")
        
        return "\n".join(relatorio)
    
    def gerar_relatorio_consolidado(self, dados_por_modelo: Dict[str, pd.DataFrame],
                                  metricas_por_modelo: Dict[str, Dict],
                                  pasta_analise: str) -> str:
        """Gera relatório consolidado final."""
        print("📝 Gerando relatório consolidado...")
        
        relatorio = []
        relatorio.append("# 📊 Relatório Consolidado de Análise de Modelos LLM")
        relatorio.append("")
        relatorio.append(f"**Data da Análise**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        
        # Estatísticas gerais
        total_respostas = sum(len(df) for df in dados_por_modelo.values())
        total_modelos = len(dados_por_modelo)
        execucoes_unicas = set()
        for df in dados_por_modelo.values():
            execucoes_unicas.update(df['execucao'].unique())
        
        # Calcular estatísticas de qualidade
        respostas_validas = 0
        for df in dados_por_modelo.values():
            if 'is_error' in df.columns:
                respostas_validas += (~df['is_error']).sum()
            else:
                respostas_validas += len(df)
        
        taxa_sucesso = (respostas_validas / total_respostas) * 100 if total_respostas > 0 else 0
        
        relatorio.append("## 📊 Informações da Análise")
        relatorio.append("")
        relatorio.append("| Métrica | Valor |")
        relatorio.append("|:--------|------:|")
        relatorio.append(f"| **Total de Respostas** | {total_respostas:,} |")
        relatorio.append(f"| **Modelos Avaliados** | {total_modelos} |")
        relatorio.append(f"| **Execuções Analisadas** | {len(execucoes_unicas)} |")
        relatorio.append(f"| **Respostas Válidas** | {respostas_validas:,} |")
        relatorio.append(f"| **Taxa de Sucesso** | {taxa_sucesso:.1f}% |")
        relatorio.append("")
        
        # Informações sobre metadados disponíveis
        primeiro_df = next(iter(dados_por_modelo.values()))
        if self._usar_campo_is_error(primeiro_df):
            relatorio.append("**Metadados**: ✅ Timestamp, comprimento de prompt/resposta, flags de erro")
        else:
            relatorio.append("**Metadados**: ⚠️ Versão anterior da pipeline (sem metadados otimizados)")
        relatorio.append("")
        
        # Resumo executivo
        relatorio.append("## 📈 Resumo Executivo")
        relatorio.append("")
        
        total_validas = sum(metricas_por_modelo[modelo]['evidently'].get('respostas_validas', 0) 
                           for modelo in dados_por_modelo.keys())
        taxa_geral = total_validas / total_respostas if total_respostas > 0 else 0
        
        # Gerar insights automáticos
        insights = self._gerar_insights_executivos(metricas_por_modelo, taxa_geral * 100)
        relatorio.extend(insights)
        relatorio.append("")
        
        # Gerar rankings detalhados
        relatorio_rankings = self._gerar_rankings_detalhados(metricas_por_modelo)
        relatorio.append(relatorio_rankings)
        
        # Ranking dos modelos (versão simplificada)
        relatorio.append("## 🏆 Ranking dos Modelos")
        relatorio.append("")
        
        modelos_ranking = self._calcular_ranking_modelos(metricas_por_modelo)
        
        for i, (modelo, score_composto) in enumerate(modelos_ranking, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}º"
            relatorio.append(f"### {emoji} {modelo} (Score: {score_composto:.4f})")
            relatorio.append("")
            
            # Métricas acadêmicas
            metricas_acad = metricas_por_modelo[modelo]['academicas']
            relatorio.append("**Métricas Acadêmicas:**")
            relatorio.append(f"- **BLEU Score**: {metricas_acad.get('bleu_medio', 0):.4f}")
            relatorio.append(f"- **ROUGE-1**: {metricas_acad.get('rouge1_medio', 0):.4f}")
            relatorio.append(f"- **ROUGE-2**: {metricas_acad.get('rouge2_medio', 0):.4f}")
            relatorio.append(f"- **ROUGE-L**: {metricas_acad.get('rougeL_medio', 0):.4f}")
            relatorio.append(f"- **BERTScore**: {metricas_acad.get('bertscore_f1_medio', 0):.4f}")
            relatorio.append("")
            
            # Métricas Evidently AI
            metricas_ev = metricas_por_modelo[modelo]['evidently']
            relatorio.append("**Métricas Evidently AI:**")
            relatorio.append(f"- **Respostas Válidas**: {metricas_ev.get('respostas_validas', 0)}")
            relatorio.append(f"- **Taxa de Validade**: {metricas_ev.get('taxa_validas', 0):.1%}")
            relatorio.append(f"- **Comprimento Médio**: {metricas_ev.get('comprimento_medio', 0):.1f} ± {metricas_ev.get('comprimento_std', 0):.1f} caracteres")
            relatorio.append(f"- **Palavras Médias**: {metricas_ev.get('palavras_medias', 0):.1f} ± {metricas_ev.get('palavras_std', 0):.1f}")
            relatorio.append("")
            
            # Métricas de Benchmarks
            if 'benchmarks' in metricas_por_modelo[modelo]:
                metricas_bench = metricas_por_modelo[modelo]['benchmarks']
                relatorio.append("**Métricas de Benchmarks:**")
                
                if 'mmlu' in metricas_bench:
                    mmlu_data = metricas_bench['mmlu']
                    relatorio.append(f"- **MMLU Accuracy**: {mmlu_data.get('accuracy', 0):.4f} ({mmlu_data.get('correct_answers', 0)}/{mmlu_data.get('total_questions', 0)})")
                
                if 'hellaswag' in metricas_bench:
                    hellaswag_data = metricas_bench['hellaswag']
                    relatorio.append(f"- **HellaSwag Accuracy**: {hellaswag_data.get('accuracy', 0):.4f} ({hellaswag_data.get('correct_answers', 0)}/{hellaswag_data.get('total_questions', 0)})")
                
                relatorio.append("")
            
            relatorio.append("---")
            relatorio.append("")
        
        # Análise comparativa
        relatorio.append("## 📊 Análise Comparativa")
        relatorio.append("")
        
        # Ranking por confiabilidade
        modelos_confiabilidade = [(modelo, metricas_por_modelo[modelo]['evidently'].get('taxa_validas', 0)) 
                                 for modelo in dados_por_modelo.keys()]
        modelos_confiabilidade.sort(key=lambda x: x[1], reverse=True)
        
        relatorio.append("**Ranking por Confiabilidade:**")
        for i, (modelo, taxa) in enumerate(modelos_confiabilidade, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}º"
            relatorio.append(f"{emoji} **{modelo}**: {taxa:.1%}")
            relatorio.append("")
            
        # Ranking por comprimento
        modelos_comprimento = [(modelo, metricas_por_modelo[modelo]['evidently'].get('comprimento_medio', 0)) 
                              for modelo in dados_por_modelo.keys()]
        modelos_comprimento.sort(key=lambda x: x[1], reverse=True)
        
        relatorio.append("**Ranking por Comprimento de Resposta:**")
        for i, (modelo, comprimento) in enumerate(modelos_comprimento, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}º"
            relatorio.append(f"{emoji} **{modelo}**: {comprimento:.1f} caracteres")
            relatorio.append("")
        
        # Recomendações
        relatorio.append("## 💡 Recomendações")
        relatorio.append("")
        
        if modelos_ranking:
            melhor_modelo = modelos_ranking[0][0]
            relatorio.append(f"### 🏆 Modelo Recomendado: {melhor_modelo}")
            relatorio.append("")
            relatorio.append("**Justificativa:**")
            relatorio.append("- Melhor score composto considerando todas as métricas")
            relatorio.append("- Equilíbrio entre precisão acadêmica e confiabilidade")
            relatorio.append("- Boa performance em métricas de qualidade textual")
            relatorio.append("")
        
        if modelos_confiabilidade:
            mais_confiavel = modelos_confiabilidade[0][0]
            relatorio.append(f"### 🛡️ Modelo Mais Confiável: {mais_confiavel}")
            relatorio.append(f"- Taxa de respostas válidas: {modelos_confiabilidade[0][1]:.1%}")
            relatorio.append("")
        
        if modelos_comprimento:
            mais_detalhado = modelos_comprimento[0][0]
            relatorio.append(f"### 📝 Modelo Mais Detalhado: {mais_detalhado}")
            relatorio.append(f"- Comprimento médio: {modelos_comprimento[0][1]:.1f} caracteres")
        relatorio.append("")
        
        return "\n".join(relatorio)
    
    def _analisar_correlacoes_metricas(self, metricas_por_modelo: Dict[str, Dict]) -> str:
        """
        Analisa correlações entre diferentes métricas para identificar consistência.
        
        Args:
            metricas_por_modelo: Dicionário com métricas por modelo
            
        Returns:
            String com análise de correlações
        """
        import numpy as np
        
        # Preparar dados para análise de correlação
        modelos = []
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bertscore_scores = []
        
        for modelo, metricas in metricas_por_modelo.items():
            metricas_acad = metricas['academicas']
            modelos.append(modelo)
            bleu_scores.append(metricas_acad.get('bleu_medio', 0))
            rouge1_scores.append(metricas_acad.get('rouge1_medio', 0))
            rouge2_scores.append(metricas_acad.get('rouge2_medio', 0))
            rougeL_scores.append(metricas_acad.get('rougeL_medio', 0))
            bertscore_scores.append(metricas_acad.get('bertscore_f1_medio', 0))
        
        # Calcular correlações
        correlacoes = []
        
        # ROUGE-1 vs BERTScore (deveria ter alta correlação)
        corr_rouge1_bert = np.corrcoef(rouge1_scores, bertscore_scores)[0, 1]
        correlacoes.append(f"- **ROUGE-1 vs BERTScore**: {corr_rouge1_bert:.3f}")
        
        # ROUGE-2 vs ROUGE-L (deveria ter alta correlação)
        corr_rouge2_rougeL = np.corrcoef(rouge2_scores, rougeL_scores)[0, 1]
        correlacoes.append(f"- **ROUGE-2 vs ROUGE-L**: {corr_rouge2_rougeL:.3f}")
        
        # BLEU vs ROUGE-1 (correlação moderada esperada)
        corr_bleu_rouge1 = np.corrcoef(bleu_scores, rouge1_scores)[0, 1]
        correlacoes.append(f"- **BLEU vs ROUGE-1**: {corr_bleu_rouge1:.3f}")
        
        # Análise de consistência
        analise = []
        analise.append("## 📊 Análise de Correlações entre Métricas")
        analise.append("")
        analise.append("### Correlações Calculadas:")
        analise.extend(correlacoes)
        analise.append("")
        
        # Interpretação
        analise.append("### Interpretação:")
        if corr_rouge1_bert > 0.7:
            analise.append("✅ **ROUGE-1 e BERTScore** têm alta correlação (consistência boa)")
        elif corr_rouge1_bert > 0.4:
            analise.append("⚠️ **ROUGE-1 e BERTScore** têm correlação moderada")
        else:
            analise.append("❌ **ROUGE-1 e BERTScore** têm baixa correlação (inconsistência)")
        
        if corr_rouge2_rougeL > 0.7:
            analise.append("✅ **ROUGE-2 e ROUGE-L** têm alta correlação (consistência boa)")
        elif corr_rouge2_rougeL > 0.4:
            analise.append("⚠️ **ROUGE-2 e ROUGE-L** têm correlação moderada")
        else:
            analise.append("❌ **ROUGE-2 e ROUGE-L** têm baixa correlação (inconsistência)")
        
        analise.append("")
        return "\n".join(analise)
    
    def _obter_descricao_metrica(self, metrica: str) -> str:
        """
        Retorna descrição amigável da métrica.
        
        Args:
            metrica: Nome da métrica
            
        Returns:
            Descrição da métrica
        """
        descricoes = {
            'BLEU': 'Mede a similaridade entre texto gerado e referência (0-1, maior é melhor)',
            'ROUGE-1': 'Mede sobreposição de palavras individuais (0-1, maior é melhor)',
            'ROUGE-2': 'Mede sobreposição de bigramas (0-1, maior é melhor)',
            'ROUGE-L': 'Mede sobreposição de subsequências mais longas (0-1, maior é melhor)',
            'BERTScore': 'Mede similaridade semântica usando embeddings BERT (0-1, maior é melhor)',
            'Taxa de Validade': 'Percentual de respostas válidas (0-1, maior é melhor)',
            'Comprimento Médio': 'Comprimento médio das respostas em caracteres',
            'Palavras Médias': 'Número médio de palavras por resposta',
            'Consistência de Comprimento': 'Consistência no tamanho das respostas (menor desvio é melhor)'
        }
        return descricoes.get(metrica, '')
    
    def _obter_descricao_categoria(self, categoria: str) -> str:
        """
        Retorna descrição amigável da categoria.
        
        Args:
            categoria: Nome da categoria
            
        Returns:
            Descrição da categoria
        """
        descricoes = {
            'Score Acadêmico': 'Combinação de métricas de qualidade de texto (BLEU, ROUGE, BERTScore)',
            'Score Evidently AI': 'Métricas de qualidade e consistência das respostas',
            'Score Geral': 'Score final combinando todas as métricas com pesos balanceados'
        }
        return descricoes.get(categoria, '')
    
    def _gerar_insights_executivos(self, metricas_por_modelo: Dict[str, Dict], taxa_sucesso: float) -> List[str]:
        """
        Gera insights executivos baseados nas métricas.
        
        Args:
            metricas_por_modelo: Dicionário com métricas por modelo
            taxa_sucesso: Taxa de sucesso geral
            
        Returns:
            Lista de insights
        """
        insights = []
        
        # Insight sobre taxa de sucesso
        if taxa_sucesso >= 90:
            insights.append("✅ **Excelente taxa de sucesso**: {:.1f}% das respostas são válidas".format(taxa_sucesso))
        elif taxa_sucesso >= 80:
            insights.append("⚠️ **Boa taxa de sucesso**: {:.1f}% das respostas são válidas".format(taxa_sucesso))
        else:
            insights.append("❌ **Taxa de sucesso baixa**: {:.1f}% das respostas são válidas".format(taxa_sucesso))
        
        # Identificar melhor modelo por categoria
        melhor_academico = None
        melhor_evidently = None
        melhor_geral = None
        
        for modelo, metricas in metricas_por_modelo.items():
            if 'academicas' in metricas:
                score_acad = metricas['academicas'].get('score_composto', 0)
                if melhor_academico is None or score_acad > melhor_academico[1]:
                    melhor_academico = (modelo, score_acad)
            
            if 'evidently' in metricas:
                score_ev = metricas['evidently'].get('taxa_validas', 0)
                if melhor_evidently is None or score_ev > melhor_evidently[1]:
                    melhor_evidently = (modelo, score_ev)
        
        if melhor_academico:
            insights.append("🏆 **Melhor modelo acadêmico**: {} (score: {:.3f})".format(
                melhor_academico[0], melhor_academico[1]))
        
        if melhor_evidently:
            insights.append("📊 **Melhor modelo em consistência**: {} (taxa: {:.1%})".format(
                melhor_evidently[0], melhor_evidently[1]))
        
        # Identificar modelos problemáticos
        modelos_problematicos = []
        for modelo, metricas in metricas_por_modelo.items():
            if 'evidently' in metricas:
                taxa_validas = metricas['evidently'].get('taxa_validas', 1)
                if taxa_validas < 0.5:  # Menos de 50% de respostas válidas
                    modelos_problematicos.append((modelo, taxa_validas))
        
        if modelos_problematicos:
            insights.append("⚠️ **Modelos com problemas**: {}".format(
                ", ".join([f"{m[0]} ({m[1]:.1%})" for m in modelos_problematicos])))
        
        return insights
    
    def _obter_emoji_rank(self, rank: int) -> str:
        """
        Retorna emoji baseado no rank.
        
        Args:
            rank: Posição no ranking
            
        Returns:
            Emoji correspondente
        """
        if rank == 1:
            return '🥇'
        elif rank == 2:
            return '🥈'
        elif rank == 3:
            return '🥉'
        elif rank <= 5:
            return '🏅'
        else:
            return '📊'
    
    def _calcular_ranking_modelos(self, metricas_por_modelo: Dict[str, Dict]) -> List[tuple]:
        """Calcula ranking dos modelos baseado em score composto."""
        rankings = []
        
        for modelo, metricas in metricas_por_modelo.items():
            # Métricas acadêmicas
            metricas_acad = metricas['academicas']
            bleu = metricas_acad.get('bleu_medio', 0)
            rouge1 = metricas_acad.get('rouge1_medio', 0)
            rouge2 = metricas_acad.get('rouge2_medio', 0)
            rougeL = metricas_acad.get('rougeL_medio', 0)
            bertscore = metricas_acad.get('bertscore_f1_medio', 0)
            
            # Métricas Evidently AI
            metricas_ev = metricas['evidently']
            taxa_validas = metricas_ev.get('taxa_validas', 0)
            respostas_validas = metricas_ev.get('respostas_validas', 0)
            
            # Penalizar modelos com poucas respostas válidas
            fator_confiabilidade = min(1.0, respostas_validas / 10.0)  # Penaliza se < 10 respostas válidas
            
            # Score composto com pesos balanceados para maior consistência
            # ROUGE-1 e BERTScore têm pesos maiores por serem mais confiáveis
            score_composto = (
                bleu * 0.10 +           # Reduzido: pode ser muito baixo
                rouge1 * 0.25 +         # Aumentado: mais confiável
                rouge2 * 0.20 +         # Aumentado: agora corrigido
                rougeL * 0.15 +         # Mantido
                bertscore * 0.25 +      # Mantido: muito confiável
                taxa_validas * 0.05     # Reduzido: não deve dominar
            ) * fator_confiabilidade
            
            # Penalização adicional para modelos com muito poucas respostas válidas
            if respostas_validas < 5:
                score_composto *= 0.3  # Penalização severa
            elif respostas_validas < 10:
                score_composto *= 0.6  # Penalização moderada
            
            rankings.append((modelo, score_composto))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def _gerar_rankings_detalhados(self, metricas_por_modelo: Dict[str, Dict]) -> str:
        """Gera rankings detalhados com métricas normalizadas."""
        print("🏆 Gerando rankings detalhados...")
        
        # Preparar dados para normalização
        dados_metricas = []
        for modelo, metricas in metricas_por_modelo.items():
            dados_modelo = {"Modelo": modelo}
            
            # Métricas acadêmicas
            metricas_acad = metricas.get('academicas', {})
            dados_modelo.update({
                "BLEU": metricas_acad.get('bleu_medio', 0),
                "ROUGE-1": metricas_acad.get('rouge1_medio', 0),
                "ROUGE-2": metricas_acad.get('rouge2_medio', 0),
                "ROUGE-L": metricas_acad.get('rougeL_medio', 0),
                "BERTScore": metricas_acad.get('bertscore_f1_medio', 0)
            })
            
            # Métricas Evidently AI
            metricas_ev = metricas.get('evidently', {})
            dados_modelo.update({
                "Respostas Válidas": metricas_ev.get('respostas_validas', 0),
                "Taxa de Validade": metricas_ev.get('taxa_validas', 0),
                "Comprimento Médio": metricas_ev.get('comprimento_medio', 0),
                "Palavras Médias": metricas_ev.get('palavras_medias', 0),
                "Consistência de Comprimento": self._calcular_consistencia_comprimento(metricas_ev)
            })
            
            dados_metricas.append(dados_modelo)
        
        # Converter para DataFrame
        df = pd.DataFrame(dados_metricas)
        
        # Normalizar métricas
        df_normalizado = self._normalizar_metricas(df)
        
        # Gerar rankings
        rankings_individuais = self._gerar_rankings_individuais(df_normalizado)
        rankings_consolidados = self._gerar_rankings_consolidados(df_normalizado)
        
        # Gerar relatório de rankings
        relatorio = []
        relatorio.append("## 🏆 Rankings Detalhados por Métrica")
        relatorio.append("")
        
        # Rankings por métrica individual
        for metrica, ranking in rankings_individuais.items():
            relatorio.append(f"### {metrica}")
            relatorio.append("")
            
            # Adicionar descrição da métrica
            descricao_metrica = self._obter_descricao_metrica(metrica)
            if descricao_metrica:
                relatorio.append(f"*{descricao_metrica}*")
                relatorio.append("")
            
            # Tabela com formatação melhorada
            relatorio.append("| 🏆 | Modelo | Score | Rank |")
            relatorio.append("|:---:|:-------|------:|:----:|")
            
            for _, row in ranking.iterrows():
                # Emoji baseado no rank
                emoji_rank = self._obter_emoji_rank(row['Rank'])
                score = row[f'Normalized {metrica}']
                relatorio.append(f"| {emoji_rank} | **{row['Modelo']}** | {score:.4f} | {row['Rank']} |")
            relatorio.append("")
        
        # Análise de correlações
        relatorio.append(self._analisar_correlacoes_metricas(metricas_por_modelo))
        relatorio.append("")
        
        # Rankings consolidados
        relatorio.append("## 📊 Rankings Consolidados por Categoria")
        relatorio.append("")
        
        for categoria, ranking in rankings_consolidados.items():
            relatorio.append(f"### {categoria}")
            relatorio.append("")
            
            # Adicionar descrição da categoria
            descricao_categoria = self._obter_descricao_categoria(categoria)
            if descricao_categoria:
                relatorio.append(f"*{descricao_categoria}*")
                relatorio.append("")
            
            # Tabela com formatação melhorada
            relatorio.append("| 🏆 | Modelo | Score | Rank |")
            relatorio.append("|:---:|:-------|------:|:----:|")
            
            for _, row in ranking.iterrows():
                emoji_rank = self._obter_emoji_rank(row['Rank'])
                score = row[categoria]
                relatorio.append(f"| {emoji_rank} | **{row['Modelo']}** | {score:.4f} | {row['Rank']} |")
            relatorio.append("")
        
        # Análise qualitativa
        analise_qualitativa = self._gerar_analise_qualitativa(df_normalizado, rankings_consolidados)
        relatorio.append(analise_qualitativa)
        
        return "\n".join(relatorio)
    
    def _calcular_consistencia_comprimento(self, metricas_ev: Dict) -> float:
        """Calcula consistência de comprimento baseada no coeficiente de variação."""
        comprimento_medio = metricas_ev.get('comprimento_medio', 0)
        comprimento_std = metricas_ev.get('comprimento_std', 0)
        
        if comprimento_medio > 0:
            cv = (comprimento_std / comprimento_medio) * 100
            # Inverter CV para ranking (menor CV = maior consistência)
            return max(0, 100 - cv)
        return 0
    
    def _normalizar_metricas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza métricas para escala 0-1 (quanto maior melhor)."""
        df_normalizado = df.copy()
        
        # Métricas acadêmicas
        academic_metrics = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]
        # Métricas Evidently AI
        evidently_metrics = ["Respostas Válidas", "Taxa de Validade", "Comprimento Médio", 
                           "Palavras Médias", "Consistência de Comprimento"]
        
        for coluna in academic_metrics + evidently_metrics:
            if coluna in df.columns:
                # Filtrar valores válidos (não nulos e não infinitos)
                valores_validos = df[coluna].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(valores_validos) == 0:
                    # Se não há valores válidos, usar 0
                    df_normalizado[f"Normalized {coluna}"] = 0.0
                    continue
                
                max_val = valores_validos.max()
                min_val = valores_validos.min()
                
                if max_val > min_val:
                    # Normalização min-max
                    df_normalizado[f"Normalized {coluna}"] = (df[coluna] - min_val) / (max_val - min_val)
                    # Garantir que valores inválidos sejam 0
                    df_normalizado[f"Normalized {coluna}"] = df_normalizado[f"Normalized {coluna}"].fillna(0.0)
                    df_normalizado[f"Normalized {coluna}"] = df_normalizado[f"Normalized {coluna}"].replace([np.inf, -np.inf], 0.0)
                else:
                    # Se todos os valores são iguais e não zero, usar 1.0
                    # Se todos são zero, usar 0.0
                    if max_val > 0:
                        df_normalizado[f"Normalized {coluna}"] = 1.0
                    else:
                        df_normalizado[f"Normalized {coluna}"] = 0.0
        
        return df_normalizado
    
    def _gerar_rankings_individuais(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Gera rankings por cada métrica individual."""
        rankings = {}
        
        # Métricas acadêmicas
        academic_metrics = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]
        # Métricas Evidently AI
        evidently_metrics = ["Respostas Válidas", "Taxa de Validade", "Comprimento Médio", 
                           "Palavras Médias", "Consistência de Comprimento"]
        
        for metrica in academic_metrics + evidently_metrics:
            coluna_normalizada = f"Normalized {metrica}"
            if coluna_normalizada in df.columns:
                ranking = df.sort_values(by=coluna_normalizada, ascending=False)[
                    ["Modelo", coluna_normalizada]
                ].reset_index(drop=True)
                ranking["Rank"] = ranking.index + 1
                rankings[metrica] = ranking
        
        return rankings
    
    def _gerar_rankings_consolidados(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Gera rankings consolidados por categoria."""
        rankings = {}
        
        # Score Acadêmico
        academic_metrics = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]
        colunas_academicas = [f"Normalized {metrica}" for metrica in academic_metrics 
                             if f"Normalized {metrica}" in df.columns]
        if colunas_academicas:
            df["Score Acadêmico"] = df[colunas_academicas].mean(axis=1)
            ranking_academico = df.sort_values(by="Score Acadêmico", ascending=False)[
                ["Modelo", "Score Acadêmico"]
            ].reset_index(drop=True)
            ranking_academico["Rank"] = ranking_academico.index + 1
            rankings["Score Acadêmico"] = ranking_academico
        
        # Score Evidently AI
        evidently_metrics = ["Respostas Válidas", "Taxa de Validade", "Comprimento Médio", 
                           "Palavras Médias", "Consistência de Comprimento"]
        colunas_evidently = [f"Normalized {metrica}" for metrica in evidently_metrics 
                            if f"Normalized {metrica}" in df.columns]
        if colunas_evidently:
            df["Score Evidently AI"] = df[colunas_evidently].mean(axis=1)
            ranking_evidently = df.sort_values(by="Score Evidently AI", ascending=False)[
                ["Modelo", "Score Evidently AI"]
            ].reset_index(drop=True)
            ranking_evidently["Rank"] = ranking_evidently.index + 1
            rankings["Score Evidently AI"] = ranking_evidently
        
        # Score Geral
        todas_colunas = colunas_academicas + colunas_evidently
        if todas_colunas:
            df["Score Geral"] = df[todas_colunas].mean(axis=1)
            ranking_geral = df.sort_values(by="Score Geral", ascending=False)[
                ["Modelo", "Score Geral"]
            ].reset_index(drop=True)
            ranking_geral["Rank"] = ranking_geral.index + 1
            rankings["Score Geral"] = ranking_geral
        
        return rankings
    
    def _gerar_analise_qualitativa(self, df: pd.DataFrame, rankings: Dict[str, pd.DataFrame]) -> str:
        """Gera análise qualitativa dos resultados."""
        analise = []
        analise.append("## 🔍 Análise Qualitativa")
        analise.append("")
        
        # Modelo mais consistente (menor variação)
        if "Normalized Consistência de Comprimento" in df.columns:
            mais_consistente = df.loc[df["Normalized Consistência de Comprimento"].idxmax(), "Modelo"]
            analise.append(f"### 🎯 Modelo Mais Consistente: {mais_consistente}")
            analise.append("- Menor variação no comprimento das respostas")
            analise.append("- Maior estabilidade de performance")
            analise.append("")
        
        # Modelo com maior fidelidade de texto (melhor BERTScore)
        if "Normalized BERTScore" in df.columns:
            melhor_bertscore = df.loc[df["Normalized BERTScore"].idxmax(), "Modelo"]
            analise.append(f"### 🧠 Modelo com Maior Fidelidade de Texto: {melhor_bertscore}")
            analise.append("- Melhor similaridade semântica com referências")
            analise.append("- Maior qualidade de conteúdo gerado")
            analise.append("")
        
        # Modelo com menor dispersão (melhor confiabilidade)
        if "Normalized Taxa de Validade" in df.columns:
            mais_confiavel = df.loc[df["Normalized Taxa de Validade"].idxmax(), "Modelo"]
            analise.append(f"### 🛡️ Modelo Mais Confiável: {mais_confiavel}")
            analise.append("- Maior taxa de respostas válidas")
            analise.append("- Menor incidência de erros")
            analise.append("")
        
        # Modelo mais detalhado (maior comprimento)
        if "Normalized Comprimento Médio" in df.columns:
            mais_detalhado = df.loc[df["Normalized Comprimento Médio"].idxmax(), "Modelo"]
            analise.append(f"### 📝 Modelo Mais Detalhado: {mais_detalhado}")
            analise.append("- Respostas mais longas e detalhadas")
            analise.append("- Maior riqueza de informação")
            analise.append("")
        
        # Análise de correlações
        analise.append("### 📈 Análise de Correlações")
        analise.append("")
        
        # Correlação entre métricas acadêmicas e Evidently AI
        colunas_academicas = [f"Normalized {metrica}" for metrica in ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]
                             if f"Normalized {metrica}" in df.columns]
        colunas_evidently = [f"Normalized {metrica}" for metrica in ["Respostas Válidas", "Taxa de Validade", "Comprimento Médio", 
                           "Palavras Médias", "Consistência de Comprimento"]
                            if f"Normalized {metrica}" in df.columns]
        
        if colunas_academicas and colunas_evidently:
            score_academico = df[colunas_academicas].mean(axis=1)
            score_evidently = df[colunas_evidently].mean(axis=1)
            correlacao = np.corrcoef(score_academico, score_evidently)[0, 1]
            
            analise.append(f"- **Correlação Acadêmico vs Evidently AI**: {correlacao:.3f}")
            
            if correlacao > 0.7:
                analise.append("  - Forte correlação positiva: modelos bons academicamente também são bons em qualidade de dados")
            elif correlacao > 0.3:
                analise.append("  - Correlação moderada: alguma relação entre métricas acadêmicas e qualidade de dados")
            else:
                analise.append("  - Correlação fraca: métricas acadêmicas e qualidade de dados são independentes")
            analise.append("")
        
        # Análise de modelos open source vs proprietários
        modelos_open_source = [m for m in df['Modelo'] if any(oss in m.lower() for oss in ['llama', 'gpt_oss', 'qwen'])]
        modelos_proprietarios = [m for m in df['Modelo'] if any(prop in m.lower() for prop in ['gemini'])]
        
        if modelos_open_source and modelos_proprietarios and "Score Geral" in df.columns:
            score_oss = df[df['Modelo'].isin(modelos_open_source)]["Score Geral"].mean()
            score_prop = df[df['Modelo'].isin(modelos_proprietarios)]["Score Geral"].mean()
            
            analise.append("### 🔓 vs 🔒 Open Source vs Proprietários")
            analise.append("")
            analise.append(f"- **Score Médio Open Source**: {score_oss:.3f}")
            analise.append(f"- **Score Médio Proprietários**: {score_prop:.3f}")
            
            if score_oss > score_prop:
                analise.append("- **Conclusão**: Modelos open source superam os proprietários em performance geral")
            elif score_prop > score_oss:
                analise.append("- **Conclusão**: Modelos proprietários superam os open source em performance geral")
            else:
                analise.append("- **Conclusão**: Performance similar entre modelos open source e proprietários")
            analise.append("")
        
        return "\n".join(analise)
    
    def _salvar_rankings_detalhados(self, metricas_por_modelo: Dict[str, Dict], pasta_analise: str):
        """Salva rankings detalhados em arquivos separados."""
        print("💾 Salvando rankings detalhados...")
        
        # Preparar dados para normalização
        dados_metricas = []
        for modelo, metricas in metricas_por_modelo.items():
            dados_modelo = {"Modelo": modelo}
            
            # Métricas acadêmicas
            metricas_acad = metricas.get('academicas', {})
            dados_modelo.update({
                "BLEU": metricas_acad.get('bleu_medio', 0),
                "ROUGE-1": metricas_acad.get('rouge1_medio', 0),
                "ROUGE-2": metricas_acad.get('rouge2_medio', 0),
                "ROUGE-L": metricas_acad.get('rougeL_medio', 0),
                "BERTScore": metricas_acad.get('bertscore_f1_medio', 0)
            })
            
            # Métricas Evidently AI
            metricas_ev = metricas.get('evidently', {})
            dados_modelo.update({
                "Respostas Válidas": metricas_ev.get('respostas_validas', 0),
                "Taxa de Validade": metricas_ev.get('taxa_validas', 0),
                "Comprimento Médio": metricas_ev.get('comprimento_medio', 0),
                "Palavras Médias": metricas_ev.get('palavras_medias', 0),
                "Consistência de Comprimento": self._calcular_consistencia_comprimento(metricas_ev)
            })
            
            dados_metricas.append(dados_modelo)
        
        # Converter para DataFrame
        df = pd.DataFrame(dados_metricas)
        
        # Normalizar métricas
        df_normalizado = self._normalizar_metricas(df)
        
        # Gerar rankings
        rankings_individuais = self._gerar_rankings_individuais(df_normalizado)
        rankings_consolidados = self._gerar_rankings_consolidados(df_normalizado)
        
        # Salvar arquivo de rankings principal
        arquivo_rankings = os.path.join(pasta_analise, "rankings.md")
        with open(arquivo_rankings, 'w', encoding='utf-8') as f:
            f.write("# 🏆 Rankings Comparativos de Modelos LLM\n\n")
            f.write(f"**Data da Análise**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
            
            # Rankings por métrica individual
            f.write("## Rankings por Métrica Individual\n\n")
            for metrica, ranking in rankings_individuais.items():
                f.write(f"### {metrica}\n")
                f.write(ranking.to_markdown(index=False))
                f.write("\n\n")
            
            # Rankings consolidados
            f.write("## Rankings Consolidados por Categoria\n\n")
            for categoria, ranking in rankings_consolidados.items():
                f.write(f"### {categoria}\n")
                f.write(ranking.to_markdown(index=False))
                f.write("\n\n")
        
        # Salvar métricas normalizadas em JSON
        arquivo_json = os.path.join(pasta_analise, "normalized_metrics.json")
        df_normalizado.to_json(arquivo_json, orient='records', indent=2, force_ascii=False)
        
        # Salvar script de geração de rankings
        script_rankings = os.path.join(pasta_analise, "generate_rankings.py")
        self._gerar_script_rankings(script_rankings)
        
        print(f"✅ Rankings salvos em: {arquivo_rankings}")
        print(f"✅ Métricas normalizadas em: {arquivo_json}")
        print(f"✅ Script de geração em: {script_rankings}")
    
    def _gerar_script_rankings(self, caminho_script: str):
        """Gera script Python para reproduzir os rankings."""
        script_content = '''import json
import pandas as pd

with open("normalized_metrics.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Define metric categories
academic_metrics = [
    "Normalized BLEU",
    "Normalized ROUGE-1",
    "Normalized ROUGE-2", 
    "Normalized ROUGE-L",
    "Normalized BERTScore"
]
evidently_ai_metrics = [
    "Normalized Respostas Válidas",
    "Normalized Taxa de Validade",
    "Normalized Comprimento Médio",
    "Normalized Palavras Médias",
    "Normalized Consistência de Comprimento"
]

# --- Ranking por cada métrica individual ---
individual_rankings = {}
for col in academic_metrics + evidently_ai_metrics:
    if col in df.columns:
        individual_rankings[col] = df.sort_values(by=col, ascending=False)[["Modelo", col]].reset_index(drop=True)
        individual_rankings[col]["Rank"] = individual_rankings[col].index + 1

# --- Ranking consolidado por categoria ---
if academic_metrics:
    df["Score Acadêmico"] = df[academic_metrics].mean(axis=1)
    academic_ranking = df.sort_values(by="Score Acadêmico", ascending=False)[[
        "Modelo", "Score Acadêmico"]].reset_index(drop=True)
    academic_ranking["Rank"] = academic_ranking.index + 1

if evidently_ai_metrics:
    df["Score Evidently AI"] = df[evidently_ai_metrics].mean(axis=1)
    evidently_ai_ranking = df.sort_values(by="Score Evidently AI", ascending=False)[[
        "Modelo", "Score Evidently AI"]].reset_index(drop=True)
    evidently_ai_ranking["Rank"] = evidently_ai_ranking.index + 1

# --- Ranking geral ---
all_metrics = academic_metrics + evidently_ai_metrics
if all_metrics:
    df["Score Geral"] = df[all_metrics].mean(axis=1)
    general_ranking = df.sort_values(by="Score Geral", ascending=False)[[
        "Modelo", "Score Geral"]].reset_index(drop=True)
    general_ranking["Rank"] = general_ranking.index + 1

# --- Salvar resultados em arquivos Markdown ---
with open("rankings.md", "w", encoding="utf-8") as f:
    f.write("# 🏆 Rankings Comparativos de Modelos LLM\\n\\n")

    f.write("## Rankings por Métrica Individual\\n\\n")
    for metric, ranking_df in individual_rankings.items():
        f.write(f"### {metric.replace('Normalized ', '')}\\n")
        f.write(ranking_df.to_markdown(index=False))
        f.write("\\n\\n")

    f.write("## Rankings Consolidados por Categoria\\n\\n")
    if 'academic_ranking' in locals():
        f.write("### Score Acadêmico\\n")
        f.write(academic_ranking.to_markdown(index=False))
        f.write("\\n\\n")
    if 'evidently_ai_ranking' in locals():
        f.write("### Score Evidently AI\\n")
        f.write(evidently_ai_ranking.to_markdown(index=False))
        f.write("\\n\\n")

    f.write("## Ranking Geral\\n\\n")
    if 'general_ranking' in locals():
        f.write(general_ranking.to_markdown(index=False))
        f.write("\\n\\n")

print("Rankings gerados e salvos em rankings.md")
'''
        
        with open(caminho_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
    
    def executar_analise_completa(self) -> str:
        """Executa análise completa e retorna caminho do relatório."""
        print("🚀 Iniciando Análise Consolidada")
        print("=" * 60)
        
        # Encontrar execuções
        execucoes = self.encontrar_execucoes()
        print(f"📁 Encontradas {len(execucoes)} execuções")
        
        if not execucoes:
            print("❌ Nenhuma execução encontrada para análise")
            return None
        
        # Consolidar dados por modelo
        dados_por_modelo = self.consolidar_dados_por_modelo(execucoes)
        
        if not dados_por_modelo:
            print("❌ Nenhum modelo encontrado para análise")
            return None
        
        # Criar pasta de análise
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pasta_analise = os.path.join(self.pasta_analysis, f"analise_consolidada_{timestamp}")
        os.makedirs(pasta_analise, exist_ok=True)
        
        # Processar cada modelo
        metricas_por_modelo = {}
        
        for modelo, df in dados_por_modelo.items():
            print(f"\n🤖 Processando modelo: {modelo}")
            
            # Criar pasta do modelo
            pasta_modelo = os.path.join(pasta_analise, f"modelo_{modelo}")
            os.makedirs(pasta_modelo, exist_ok=True)
            
            # Calcular métricas acadêmicas
            df_com_metricas, metricas_academicas, relatorio_academicas = self.calcular_metricas_academicas(df)
            
            # Calcular métricas Evidently AI
            metricas_evidently = self.calcular_metricas_evidently(df_com_metricas)
            
            # Calcular métricas de benchmarks
            metricas_benchmarks = self.calcular_metricas_benchmarks(df_com_metricas)
            
            # Gerar relatórios Evidently AI
            try:
                # Tentar import relativo (quando chamado via main.py)
                from .evidently_reports import gerar_relatorios_evidently_completo
            except ImportError:
                # Fallback para import absoluto (quando executado diretamente)
                from evidently_reports import gerar_relatorios_evidently_completo
            
            try:
                pasta_evidently = os.path.join(pasta_modelo, "evidently_reports")
                relatorios_evidently, relatorio_evidently = gerar_relatorios_evidently_completo(df_com_metricas, pasta_evidently)
                print(f"📊 Relatórios Evidently AI salvos em: {pasta_evidently}")
            except Exception as e:
                print(f"⚠️ Erro ao gerar relatórios Evidently AI: {e}")
                relatorios_evidently = {}
                relatorio_evidently = ""
            
            # Gerar relatório individual do modelo
            relatorio_modelo = self.gerar_relatorio_por_modelo(modelo, df_com_metricas, 
                                                              metricas_academicas, metricas_evidently, 
                                                              metricas_benchmarks, pasta_modelo)
            
            # Adicionar relatório Evidently AI
            if relatorio_evidently:
                relatorio_modelo += "\n\n" + relatorio_evidently
            
            # Adicionar análise do HTML do Evidently AI
            pasta_evidently = os.path.join(pasta_modelo, "evidently_reports")
            if os.path.exists(pasta_evidently):
                relatorio_html = self._analisar_html_evidently(pasta_evidently)
                relatorio_modelo += "\n\n" + relatorio_html
            
            # Adicionar métricas detalhadas do Evidently AI baseadas nos dados
            relatorio_detalhado = self._gerar_metricas_detalhadas_evidently(df_com_metricas)
            relatorio_modelo += "\n\n" + relatorio_detalhado
            
            # Salvar relatório do modelo
            arquivo_relatorio_modelo = os.path.join(pasta_modelo, f"relatorio_{modelo}.md")
            with open(arquivo_relatorio_modelo, 'w', encoding='utf-8') as f:
                f.write(relatorio_modelo)
            
            # Salvar dados do modelo
            df_com_metricas.to_csv(os.path.join(pasta_modelo, f"dados_{modelo}.csv"), 
                                  index=False, encoding=self.config.ENCODING_CSV)
            
            # Armazenar métricas
            metricas_por_modelo[modelo] = {
                'academicas': metricas_academicas,
                'evidently': metricas_evidently,
                'benchmarks': metricas_benchmarks.get(modelo, {})
            }
            
            print(f"✅ {modelo}: Relatório salvo em {arquivo_relatorio_modelo}")
        
        # Gerar relatório consolidado
        relatorio_consolidado = self.gerar_relatorio_consolidado(dados_por_modelo, metricas_por_modelo, pasta_analise)
        
        # Salvar relatório consolidado
        arquivo_relatorio_consolidado = os.path.join(pasta_analise, "relatorio_consolidado.md")
        with open(arquivo_relatorio_consolidado, 'w', encoding='utf-8') as f:
            f.write(relatorio_consolidado)
        
        # Salvar métricas consolidadas
        with open(os.path.join(pasta_analise, "metricas_consolidadas.json"), 'w', encoding='utf-8') as f:
            json.dump(metricas_por_modelo, f, indent=2, ensure_ascii=False, default=str)
        
        # Gerar e salvar rankings detalhados
        self._salvar_rankings_detalhados(metricas_por_modelo, pasta_analise)
        
        print(f"\n💾 Análise consolidada salva em: {pasta_analise}")
        print(f"📄 Relatório consolidado: {arquivo_relatorio_consolidado}")
        
        return arquivo_relatorio_consolidado

def executar_analise():
    """Função principal para executar análise consolidada."""
    analyzer = AnalysisSystem()
    return analyzer.executar_analise_completa()

if __name__ == "__main__":
    print("🚀 Executando análise consolidada diretamente...")
    print("=" * 60)
    
    try:
        resultado = executar_analise()
        if resultado:
            print(f"\n✅ Análise concluída com sucesso!")
            print(f"📄 Relatório salvo em: {resultado}")
        else:
            print("\n❌ Análise não pôde ser concluída")
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()