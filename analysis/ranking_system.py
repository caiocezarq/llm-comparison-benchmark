#!/usr/bin/env python3
"""
Sistema de Ranking Comparativo de Modelos LLM
Extrai métricas dos relatórios .md e gera rankings consolidados.
"""

import pandas as pd
import numpy as np
import os
import re
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys

# Adicionar o diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config

class RankingSystem:
    """Sistema de ranking comparativo de modelos LLM."""
    
    def __init__(self):
        self.config = get_config()
        self.pasta_analysis = "analysis"
        
        # Métricas acadêmicas
        self.academic_metrics = [
            "BLEU",
            "ROUGE-1", 
            "ROUGE-2",
            "ROUGE-L",
            "BERTScore"
        ]
        
        # Métricas Evidently AI
        self.evidently_metrics = [
            "Respostas Válidas",
            "Taxa de Validade", 
            "Comprimento Médio",
            "Palavras Médias",
            "Consistência de Comprimento"
        ]
        
        # Métricas de Benchmarks
        self.benchmark_metrics = [
            "MMLU Accuracy",
            "HellaSwag Accuracy"
        ]
    
    def extrair_metricas_de_relatorios(self, pasta_analise: str) -> Dict[str, Dict]:
        """
        Extrai métricas dos relatórios .md de cada modelo.
        
        Args:
            pasta_analise: Caminho da pasta de análise consolidada
            
        Returns:
            Dicionário com métricas por modelo
        """
        print("📊 Extraindo métricas dos relatórios .md...")
        
        metricas_por_modelo = {}
        
        # Encontrar pastas de modelos
        for item in os.listdir(pasta_analise):
            if item.startswith("modelo_"):
                modelo = item.replace("modelo_", "")
                pasta_modelo = os.path.join(pasta_analise, item)
                
                # Procurar relatório do modelo
                arquivo_relatorio = os.path.join(pasta_modelo, f"relatorio_{modelo}.md")
                
                if os.path.exists(arquivo_relatorio):
                    print(f"🔍 Processando {modelo}...")
                    metricas = self._extrair_metricas_do_relatorio(arquivo_relatorio)
                    if metricas:
                        metricas_por_modelo[modelo] = metricas
                        print(f"✅ {modelo}: {len(metricas)} métricas extraídas")
                    else:
                        print(f"⚠️ {modelo}: Nenhuma métrica extraída")
                else:
                    print(f"❌ {modelo}: Relatório não encontrado")
        
        return metricas_por_modelo
    
    def extrair_metricas_benchmarks(self, pasta_analise: str) -> Dict[str, Dict]:
        """
        Extrai métricas de benchmarks dos arquivos JSON.
        
        Args:
            pasta_analise: Caminho da pasta de análise
            
        Returns:
            Dicionário com métricas de benchmarks por modelo
        """
        metricas_benchmarks = {}
        
        # Procurar arquivo de métricas consolidadas
        arquivo_metricas = os.path.join(pasta_analise, "metricas_consolidadas.json")
        
        if not os.path.exists(arquivo_metricas):
            print(f"⚠️ Arquivo {arquivo_metricas} não encontrado")
            return {}
        
        try:
            with open(arquivo_metricas, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            
            # Extrair métricas de benchmarks
            for modelo, metricas in dados.items():
                if 'benchmarks' in metricas:
                    metricas_benchmarks[modelo] = {}
                    
                    # MMLU
                    if 'mmlu' in metricas['benchmarks']:
                        mmlu_data = metricas['benchmarks']['mmlu']
                        metricas_benchmarks[modelo]['MMLU Accuracy'] = mmlu_data.get('accuracy_valid_only', mmlu_data.get('accuracy', 0.0))
                        metricas_benchmarks[modelo]['MMLU Coverage'] = mmlu_data.get('coverage', 0.0)
                        metricas_benchmarks[modelo]['MMLU Total Questions'] = mmlu_data.get('total_questions', 0)
                        metricas_benchmarks[modelo]['MMLU Correct Answers'] = mmlu_data.get('correct_answers', 0)
                    
                    # HellaSwag
                    if 'hellaswag' in metricas['benchmarks']:
                        hellaswag_data = metricas['benchmarks']['hellaswag']
                        metricas_benchmarks[modelo]['HellaSwag Accuracy'] = hellaswag_data.get('accuracy_valid_only', hellaswag_data.get('accuracy', 0.0))
                        metricas_benchmarks[modelo]['HellaSwag Coverage'] = hellaswag_data.get('coverage', 0.0)
                        metricas_benchmarks[modelo]['HellaSwag Total Questions'] = hellaswag_data.get('total_questions', 0)
                        metricas_benchmarks[modelo]['HellaSwag Correct Answers'] = hellaswag_data.get('correct_answers', 0)
        
        except Exception as e:
            print(f"❌ Erro ao extrair métricas de benchmarks: {e}")
            return {}
        
        return metricas_benchmarks
    
    def _extrair_metricas_do_relatorio(self, arquivo_relatorio: str) -> Dict[str, float]:
        """
        Extrai métricas específicas de um relatório .md.
        
        Args:
            arquivo_relatorio: Caminho do arquivo .md
            
        Returns:
            Dicionário com métricas extraídas
        """
        try:
            with open(arquivo_relatorio, 'r', encoding='utf-8') as f:
                conteudo = f.read()
            
            metricas = {}
            
            # Padrões para extrair métricas acadêmicas
            padroes_academicas = {
                "BLEU": r"BLEU Score[:\s]*([0-9.]+)",
                "ROUGE-1": r"ROUGE-1[:\s]*([0-9.]+)",
                "ROUGE-2": r"ROUGE-2[:\s]*([0-9.]+)", 
                "ROUGE-L": r"ROUGE-L[:\s]*([0-9.]+)",
                "BERTScore": r"BERTScore[:\s]*([0-9.]+)"
            }
            
            # Padrões para extrair métricas Evidently AI
            padroes_evidently = {
                "Respostas Válidas": r"Respostas Válidas[:\s]*([0-9]+)",
                "Taxa de Validade": r"Taxa de Validade[:\s]*([0-9.]+)%?",
                "Comprimento Médio": r"Comprimento Médio[:\s]*([0-9.]+)",
                "Palavras Médias": r"Palavras Médias[:\s]*([0-9.]+)",
                "Consistência de Comprimento": r"Consistência de Comprimento[:\s]*([0-9.]+)%?"
            }
            
            # Extrair métricas acadêmicas
            for metrica, padrao in padroes_academicas.items():
                match = re.search(padrao, conteudo, re.IGNORECASE)
                if match:
                    try:
                        valor = float(match.group(1))
                        metricas[metrica] = valor
                    except ValueError:
                        print(f"⚠️ Erro ao converter {metrica}: {match.group(1)}")
            
            # Extrair métricas Evidently AI
            for metrica, padrao in padroes_evidently.items():
                match = re.search(padrao, conteudo, re.IGNORECASE)
                if match:
                    try:
                        valor = float(match.group(1))
                        # Converter percentual para decimal se necessário
                        if metrica == "Taxa de Validade" and valor > 1:
                            valor = valor / 100
                        metricas[metrica] = valor
                    except ValueError:
                        print(f"⚠️ Erro ao converter {metrica}: {match.group(1)}")
            
            # Calcular consistência de comprimento se não encontrada diretamente
            if "Consistência de Comprimento" not in metricas:
                # Tentar calcular a partir do CV se disponível
                cv_match = re.search(r"CV[:\s]*([0-9.]+)%", conteudo, re.IGNORECASE)
                if cv_match:
                    try:
                        cv = float(cv_match.group(1))
                        # Inverter CV para ranking (menor CV = maior consistência)
                        metricas["Consistência de Comprimento"] = max(0, 100 - cv)
                    except ValueError:
                        pass
            
            return metricas
            
        except Exception as e:
            print(f"❌ Erro ao extrair métricas de {arquivo_relatorio}: {e}")
            return {}
    
    def normalizar_metricas(self, metricas_por_modelo: Dict[str, Dict]) -> pd.DataFrame:
        """
        Normaliza métricas para escala 0-1 (quanto maior melhor).
        
        Args:
            metricas_por_modelo: Dicionário com métricas por modelo
            
        Returns:
            DataFrame com métricas normalizadas
        """
        print("🔄 Normalizando métricas...")
        
        # Converter para DataFrame
        df = pd.DataFrame.from_dict(metricas_por_modelo, orient='index')
        
        # Preencher valores ausentes com 0
        df = df.fillna(0)
        
        # Normalizar métricas (quanto maior melhor)
        for coluna in df.columns:
            if coluna in self.academic_metrics + self.evidently_metrics + self.benchmark_metrics:
                # Filtrar valores válidos (não nulos e não infinitos)
                valores_validos = df[coluna].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(valores_validos) == 0:
                    # Se não há valores válidos, usar 0
                    df[f"Normalized {coluna}"] = 0.0
                    continue
                
                max_val = valores_validos.max()
                min_val = valores_validos.min()
                
                if max_val > min_val:
                    # Normalização min-max
                    df[f"Normalized {coluna}"] = (df[coluna] - min_val) / (max_val - min_val)
                    # Garantir que valores inválidos sejam 0
                    df[f"Normalized {coluna}"] = df[f"Normalized {coluna}"].fillna(0.0)
                    df[f"Normalized {coluna}"] = df[f"Normalized {coluna}"].replace([np.inf, -np.inf], 0.0)
                else:
                    # Se todos os valores são iguais e não zero, usar 1.0
                    # Se todos são zero, usar 0.0
                    if max_val > 0:
                        df[f"Normalized {coluna}"] = 1.0
                    else:
                        df[f"Normalized {coluna}"] = 0.0
        
        return df
    
    def gerar_rankings_individuais(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Gera rankings por cada métrica individual.
        
        Args:
            df: DataFrame com métricas normalizadas
            
        Returns:
            Dicionário com rankings por métrica
        """
        print("🏆 Gerando rankings individuais...")
        
        rankings = {}
        
        # Rankings por métricas acadêmicas
        for metrica in self.academic_metrics:
            coluna_normalizada = f"Normalized {metrica}"
            if coluna_normalizada in df.columns:
                ranking = df.sort_values(by=coluna_normalizada, ascending=False)[
                    ["Modelo", coluna_normalizada]
                ].reset_index(drop=True)
                ranking["Rank"] = ranking.index + 1
                rankings[metrica] = ranking
        
        # Rankings por métricas Evidently AI
        for metrica in self.evidently_metrics:
            coluna_normalizada = f"Normalized {metrica}"
            if coluna_normalizada in df.columns:
                ranking = df.sort_values(by=coluna_normalizada, ascending=False)[
                    ["Modelo", coluna_normalizada]
                ].reset_index(drop=True)
                ranking["Rank"] = ranking.index + 1
                rankings[metrica] = ranking
        
        return rankings
    
    def gerar_rankings_consolidados(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Gera rankings consolidados por categoria.
        
        Args:
            df: DataFrame com métricas normalizadas
            
        Returns:
            Dicionário com rankings consolidados
        """
        print("📊 Gerando rankings consolidados...")
        
        rankings = {}
        
        # Score Acadêmico
        colunas_academicas = [f"Normalized {metrica}" for metrica in self.academic_metrics 
                             if f"Normalized {metrica}" in df.columns]
        if colunas_academicas:
            df["Score Acadêmico"] = df[colunas_academicas].mean(axis=1)
            ranking_academico = df.sort_values(by="Score Acadêmico", ascending=False)[
                ["Modelo", "Score Acadêmico"]
            ].reset_index(drop=True)
            ranking_academico["Rank"] = ranking_academico.index + 1
            rankings["Score Acadêmico"] = ranking_academico
        
        # Score Evidently AI
        colunas_evidently = [f"Normalized {metrica}" for metrica in self.evidently_metrics 
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
    
    def gerar_analise_qualitativa(self, df: pd.DataFrame, rankings: Dict[str, pd.DataFrame]) -> str:
        """
        Gera análise qualitativa dos resultados.
        
        Args:
            df: DataFrame com métricas normalizadas
            rankings: Dicionário com rankings
            
        Returns:
            String com análise qualitativa
        """
        print("📝 Gerando análise qualitativa...")
        
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
        colunas_academicas = [f"Normalized {metrica}" for metrica in self.academic_metrics 
                             if f"Normalized {metrica}" in df.columns]
        colunas_evidently = [f"Normalized {metrica}" for metrica in self.evidently_metrics 
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
        
        # Recomendações finais
        analise.append("### 💡 Recomendações")
        analise.append("")
        
        if "Score Geral" in df.columns:
            melhor_geral = df.loc[df["Score Geral"].idxmax(), "Modelo"]
            analise.append(f"**🏆 Modelo Recomendado**: {melhor_geral}")
            analise.append("- Melhor score geral considerando todas as métricas")
            analise.append("- Equilíbrio entre qualidade acadêmica e confiabilidade")
            analise.append("")
        
        # Análise de modelos open source vs proprietários
        modelos_open_source = [m for m in df.index if any(oss in m.lower() for oss in ['llama', 'gpt_oss', 'qwen'])]
        modelos_proprietarios = [m for m in df.index if any(prop in m.lower() for prop in ['gemini'])]
        
        if modelos_open_source and modelos_proprietarios:
            score_oss = df.loc[modelos_open_source, "Score Geral"].mean() if "Score Geral" in df.columns else 0
            score_prop = df.loc[modelos_proprietarios, "Score Geral"].mean() if "Score Geral" in df.columns else 0
            
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
    
    def salvar_rankings(self, rankings_individuais: Dict[str, pd.DataFrame], 
                       rankings_consolidados: Dict[str, pd.DataFrame],
                       df_normalizado: pd.DataFrame,
                       analise_qualitativa: str,
                       pasta_destino: str) -> str:
        """
        Salva todos os rankings em arquivos Markdown e JSON.
        
        Args:
            rankings_individuais: Rankings por métrica individual
            rankings_consolidados: Rankings consolidados
            df_normalizado: DataFrame com métricas normalizadas
            analise_qualitativa: Análise qualitativa
            pasta_destino: Pasta de destino
            
        Returns:
            Caminho do arquivo principal de rankings
        """
        print("💾 Salvando rankings...")
        
        # Criar pasta de destino se não existir
        os.makedirs(pasta_destino, exist_ok=True)
        
        # Arquivo principal de rankings
        arquivo_rankings = os.path.join(pasta_destino, "rankings.md")
        
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
            
            # Análise qualitativa
            f.write(analise_qualitativa)
        
        # Salvar métricas normalizadas em JSON
        arquivo_json = os.path.join(pasta_destino, "normalized_metrics.json")
        df_normalizado.to_json(arquivo_json, orient='records', indent=2, force_ascii=False)
        
        # Salvar script de geração de rankings
        script_rankings = os.path.join(pasta_destino, "generate_rankings.py")
        self._gerar_script_rankings(script_rankings)
        
        print(f"✅ Rankings salvos em: {arquivo_rankings}")
        print(f"✅ Métricas normalizadas em: {arquivo_json}")
        print(f"✅ Script de geração em: {script_rankings}")
        
        return arquivo_rankings
    
    def _gerar_script_rankings(self, caminho_script: str):
        """
        Gera script Python para reproduzir os rankings.
        
        Args:
            caminho_script: Caminho do arquivo de script
        """
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
    
    def executar_analise_ranking(self, pasta_analise: str) -> str:
        """
        Executa análise completa de ranking.
        
        Args:
            pasta_analise: Pasta de análise consolidada
            
        Returns:
            Caminho do arquivo de rankings gerado
        """
        print("🚀 Iniciando Análise de Ranking")
        print("=" * 60)
        
        # Extrair métricas dos relatórios
        metricas_por_modelo = self.extrair_metricas_de_relatorios(pasta_analise)
        
        # Extrair métricas de benchmarks
        metricas_benchmarks = self.extrair_metricas_benchmarks(pasta_analise)
        
        if not metricas_por_modelo:
            print("❌ Nenhuma métrica extraída dos relatórios")
            return None
        
        # Combinar métricas acadêmicas e benchmarks
        metricas_combinadas = {}
        for modelo in metricas_por_modelo:
            metricas_combinadas[modelo] = metricas_por_modelo[modelo].copy()
            if modelo in metricas_benchmarks:
                metricas_combinadas[modelo].update(metricas_benchmarks[modelo])
        
        # Normalizar métricas
        df_normalizado = self.normalizar_metricas(metricas_combinadas)
        
        # Gerar rankings individuais
        rankings_individuais = self.gerar_rankings_individuais(df_normalizado)
        
        # Gerar rankings consolidados
        rankings_consolidados = self.gerar_rankings_consolidados(df_normalizado)
        
        # Gerar análise qualitativa
        analise_qualitativa = self.gerar_analise_qualitativa(df_normalizado, rankings_consolidados)
        
        # Salvar resultados
        pasta_rankings = os.path.join(pasta_analise, "rankings")
        arquivo_rankings = self.salvar_rankings(
            rankings_individuais, 
            rankings_consolidados,
            df_normalizado,
            analise_qualitativa,
            pasta_rankings
        )
        
        print(f"\n💾 Análise de ranking salva em: {pasta_rankings}")
        print(f"📄 Arquivo principal: {arquivo_rankings}")
        
        return arquivo_rankings

def executar_ranking(pasta_analise: str) -> str:
    """
    Função principal para executar análise de ranking.
    
    Args:
        pasta_analise: Pasta de análise consolidada
        
    Returns:
        Caminho do arquivo de rankings gerado
    """
    ranking_system = RankingSystem()
    return ranking_system.executar_analise_ranking(pasta_analise)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pasta_analise = sys.argv[1]
    else:
        # Procurar pasta de análise mais recente
        pasta_analysis = "analysis"
        if os.path.exists(pasta_analysis):
            pastas = [d for d in os.listdir(pasta_analysis) if d.startswith("analise_consolidada_")]
            if pastas:
                pasta_analise = os.path.join(pasta_analysis, sorted(pastas)[-1])
            else:
                print("❌ Nenhuma pasta de análise encontrada")
                sys.exit(1)
        else:
            print("❌ Pasta analysis não encontrada")
            sys.exit(1)
    
    executar_ranking(pasta_analise)
