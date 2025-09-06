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

class AnalysisSystem:
    """Sistema principal de análise consolidada."""
    
    def __init__(self):
        self.config = get_config()
        self.pasta_analysis = "analysis"
        self.pasta_resultados = self.config.PASTA_RESULTADOS
        self.prefixo_execucao = self.config.PREFIXO_EXECUCAO
    
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
            from .bleu_rouge import calcular_bleu_rouge_completo
            df_bleu_rouge, metricas_bleu_rouge, relatorio_bleu_rouge = calcular_bleu_rouge_completo(df)
        except Exception as e:
            print(f"❌ Erro ao calcular BLEU/ROUGE: {e}")
            return df, {}, "Erro ao calcular BLEU/ROUGE"
        
        # BERTScore
        try:
            from .bertscore import calcular_bertscore_completo
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
    
    def _calcular_metricas_agregadas(self, df: pd.DataFrame) -> Dict:
        """Calcula métricas agregadas por modelo."""
        if 'model' not in df.columns:
            return {}

        metricas_agregadas = {}

        for modelo in df['model'].unique():
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
    
    def _usar_campo_is_error(self, df: pd.DataFrame) -> bool:
        """Verifica se o DataFrame tem o campo is_error (nova versão da pipeline)."""
        return 'is_error' in df.columns
    
    def gerar_relatorio_por_modelo(self, modelo: str, df: pd.DataFrame, 
                                 metricas_academicas: Dict, metricas_evidently: Dict,
                                 pasta_destino: str) -> str:
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
        
        relatorio.append(f"**Total de Respostas**: {total_respostas}")
        relatorio.append(f"**Modelos Avaliados**: {total_modelos}")
        relatorio.append(f"**Execuções Analisadas**: {len(execucoes_unicas)}")
        relatorio.append(f"**Execuções**: {', '.join(sorted(execucoes_unicas))}")
        
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
        
        relatorio.append(f"- **Respostas Válidas**: {total_validas}")
        relatorio.append(f"- **Taxa de Sucesso Geral**: {taxa_geral:.1%}")
        relatorio.append("")
        
        # Ranking dos modelos
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
            
            # Score composto com penalização por baixa confiabilidade
            score_composto = (
                bleu * 0.15 +
                rouge1 * 0.20 +
                rouge2 * 0.15 +
                rougeL * 0.15 +
                bertscore * 0.25 +
                taxa_validas * 0.10
            ) * fator_confiabilidade
            
            # Penalização adicional para modelos com muito poucas respostas válidas
            if respostas_validas < 5:
                score_composto *= 0.3  # Penalização severa
            elif respostas_validas < 10:
                score_composto *= 0.6  # Penalização moderada
            
            rankings.append((modelo, score_composto))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
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
            
            # Gerar relatórios Evidently AI
            try:
                from .evidently_reports import gerar_relatorios_evidently_completo
                pasta_evidently = os.path.join(pasta_modelo, "evidently_reports")
                relatorios_evidently, relatorio_evidently = gerar_relatorios_evidently_completo(df_com_metricas, pasta_evidently)
                print(f"📊 Relatórios Evidently AI salvos em: {pasta_evidently}")
            except Exception as e:
                print(f"⚠️ Erro ao gerar relatórios Evidently AI: {e}")
                relatorios_evidently = {}
                relatorio_evidently = ""
            
            # Gerar relatório individual do modelo
            relatorio_modelo = self.gerar_relatorio_por_modelo(modelo, df_com_metricas, 
                                                              metricas_academicas, metricas_evidently, pasta_modelo)
            
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
                'evidently': metricas_evidently
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
        
        print(f"\n💾 Análise consolidada salva em: {pasta_analise}")
        print(f"📄 Relatório consolidado: {arquivo_relatorio_consolidado}")
        
        return arquivo_relatorio_consolidado

def executar_analise():
    """Função principal para executar análise consolidada."""
    analyzer = AnalysisSystem()
    return analyzer.executar_analise_completa()

if __name__ == "__main__":
    executar_analise()