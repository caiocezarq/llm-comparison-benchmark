import json
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
    f.write("# 🏆 Rankings Comparativos de Modelos LLM\n\n")

    f.write("## Rankings por Métrica Individual\n\n")
    for metric, ranking_df in individual_rankings.items():
        f.write(f"### {metric.replace('Normalized ', '')}\n")
        f.write(ranking_df.to_markdown(index=False))
        f.write("\n\n")

    f.write("## Rankings Consolidados por Categoria\n\n")
    if 'academic_ranking' in locals():
        f.write("### Score Acadêmico\n")
        f.write(academic_ranking.to_markdown(index=False))
        f.write("\n\n")
    if 'evidently_ai_ranking' in locals():
        f.write("### Score Evidently AI\n")
        f.write(evidently_ai_ranking.to_markdown(index=False))
        f.write("\n\n")

    f.write("## Ranking Geral\n\n")
    if 'general_ranking' in locals():
        f.write(general_ranking.to_markdown(index=False))
        f.write("\n\n")

print("Rankings gerados e salvos em rankings.md")
