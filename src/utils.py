# utils.py
"""
Funções auxiliares para salvar resultados e carregar prompts.
"""
import json
import pandas as pd
import os
import warnings
from datetime import datetime
from .config import get_config

# Suprimir avisos desnecessários
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")

# Carregar configurações
config = get_config()

def get_next_result_folder():
    """
    Cria pasta de resultado numerada na pasta results/ (resultado_1, resultado_2, etc.)
    """
    # Verificar se a pasta results existe, se não, criar
    if not os.path.exists(config.PASTA_RESULTADOS):
        os.makedirs(config.PASTA_RESULTADOS)
    
    base_name = f"{config.PASTA_RESULTADOS}/{config.PREFIXO_EXECUCAO}"
    counter = 1
    
    while os.path.exists(f"{base_name}_{counter}"):
        counter += 1
    
    folder_name = f"{base_name}_{counter}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def save_results_csv(df, path):
    """
    Salva DataFrame em arquivo CSV com encoding UTF-8 para caracteres especiais.
    """
    encoding_config = config.get_encoding_config()
    df.to_csv(path, index=False, encoding=encoding_config['csv'])

def save_results_json(df, path):
    """
    Salva DataFrame em arquivo JSON (records) com encoding UTF-8.
    """
    # Salvar JSON sem encoding (pandas 2.0+ não suporta encoding no to_json)
    df.to_json(path, orient="records", lines=True, force_ascii=False)

def save_analysis_files(df, metricas, folder_path):
    """
    Salva arquivos de análise adicionais para trabalhos acadêmicos.
    """
    # Análise comparativa por modelo
    analysis_data = []
    for modelo in df["model"].unique():
        df_modelo = df[df["model"] == modelo]
        
        # Estatísticas básicas
        tempo_medio = df_modelo["time"].mean()
        tempo_std = df_modelo["time"].std()
        total_prompts = len(df_modelo)
        
        # Análise por idioma
        prompts_pt = df_modelo[df_modelo['prompt'].str.contains('é|como|quais|explique|descreva', case=False)]
        prompts_en = df_modelo[~df_modelo['prompt'].str.contains('é|como|quais|explique|descreva', case=False)]
        
        # Análise por domínio
        tech_pt = df_modelo[df_modelo['prompt'].str.contains('inteligência artificial|machine learning|GPT|LLaMA|computação quântica', case=False)]
        science_pt = df_modelo[df_modelo['prompt'].str.contains('mudanças climáticas|fotossíntese|evolução|energia renovável|sustentabilidade', case=False)]
        business_en = df_modelo[df_modelo['prompt'].str.contains('digital transformation|disruptive innovation|customer experience|remote work|e-commerce', case=False)]
        tech_en = df_modelo[df_modelo['prompt'].str.contains('quantum computing|neural networks|renewable energy|5G|blockchain', case=False)]
        
        analysis_data.append({
            "modelo": modelo,
            "tempo_medio_segundos": tempo_medio,
            "tempo_desvio_padrao": tempo_std,
            "total_prompts": total_prompts,
            "prompts_portugues": len(prompts_pt),
            "prompts_ingles": len(prompts_en),
            "prompts_tecnologia_pt": len(tech_pt),
            "prompts_ciencia_pt": len(science_pt),
            "prompts_business_en": len(business_en),
            "prompts_tecnologia_en": len(tech_en)
        })
    
    # Salva análise comparativa
    analysis_df = pd.DataFrame(analysis_data)
    analysis_path = os.path.join(folder_path, "analise_comparativa.csv")
    save_results_csv(analysis_df, analysis_path)
    
    # Ranking dos modelos por métricas
    if metricas:
        ranking_data = []
        for modelo, metrica in metricas.items():
            bleu_score = metrica['bleu']['bleu']
            rouge1_score = metrica['rouge']['rouge1']
            rouge2_score = metrica['rouge']['rouge2']
            rougeL_score = metrica['rouge']['rougeL']
            bert_f1 = sum(metrica['bertscore']['f1'])/len(metrica['bertscore']['f1'])
            
            # Score normalizado (0-1)
            score_normalizado = (bleu_score + rouge1_score + bert_f1) / 3
            
            ranking_data.append({
                "modelo": modelo,
                "bleu_score": bleu_score,
                "rouge1_score": rouge1_score,
                "rouge2_score": rouge2_score,
                "rougeL_score": rougeL_score,
                "bertscore_f1": bert_f1,
                "score_normalizado": score_normalizado,
                "ranking_geral": 0  # Será preenchido depois
            })
        
        # Adiciona ranking
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('score_normalizado', ascending=False)
        ranking_df['ranking_geral'] = range(1, len(ranking_df) + 1)
        
        ranking_path = os.path.join(folder_path, "ranking_modelos.csv")
        save_results_csv(ranking_df, ranking_path)
    
    return analysis_path

def generate_final_report(df, metricas, relatorios, execution_time, folder_path):
    """
    Gera relatório final consolidado com todos os resultados.
    Relatório formatado para trabalhos acadêmicos.
    """
    # Análise por domínio dos prompts
    prompts_pt_tech = df[df['prompt'].str.contains('inteligência artificial|machine learning|GPT|LLaMA|computação quântica', case=False)]
    prompts_pt_science = df[df['prompt'].str.contains('mudanças climáticas|fotossíntese|evolução|energia renovável|sustentabilidade', case=False)]
    prompts_en_business = df[df['prompt'].str.contains('digital transformation|disruptive innovation|customer experience|remote work|e-commerce', case=False)]
    prompts_en_tech = df[df['prompt'].str.contains('quantum computing|neural networks|renewable energy|5G|blockchain', case=False)]
    
    # Análise por idioma
    prompts_portuguese = df[df['prompt'].str.contains('é|como|quais|explique|descreva', case=False)]
    prompts_english = df[~df['prompt'].str.contains('é|como|quais|explique|descreva', case=False)]
    
    report = {
        "execucao": {
            "data_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tempo_execucao": execution_time,
            "total_respostas": len(df),
            "modelos_testados": df["model"].unique().tolist(),
            "prompts_executados": len(df["prompt"].unique()),
            "prompts_por_idioma": {
                "portugues": len(prompts_portuguese),
                "ingles": len(prompts_english)
            },
            "prompts_por_dominio": {
                "tecnologia_pt": len(prompts_pt_tech),
                "ciencia_pt": len(prompts_pt_science),
                "business_en": len(prompts_en_business),
                "tecnologia_en": len(prompts_en_tech)
            }
        },
        "metricas_por_modelo": metricas,
        "estatisticas_gerais": {
            "tempo_medio_por_modelo": df.groupby("model")["time"].mean().to_dict(),
            "tempo_total_por_modelo": df.groupby("model")["time"].sum().to_dict(),
            "modelo_mais_rapido": df.groupby("model")["time"].mean().idxmin(),
            "modelo_mais_lento": df.groupby("model")["time"].mean().idxmax(),
            "desvio_padrao_tempo": df.groupby("model")["time"].std().to_dict()
        },
        "analise_por_dominio": {},
        "ranking_modelos": {},
        "evidently_disponivel": bool(relatorios),
        "arquivos_gerados": [
            "resultados_todos.csv",
            "resultados_todos.json",
            "analise_comparativa.csv",
            "ranking_modelos.csv"
        ] + [f"resultados_{model}.csv" for model in df["model"].unique()]
    }
    
    # Análise por domínio
    for dominio, df_dominio in [("Tecnologia PT", prompts_pt_tech), ("Ciência PT", prompts_pt_science), 
                                ("Business EN", prompts_en_business), ("Tecnologia EN", prompts_en_tech)]:
        if len(df_dominio) > 0:
            report["analise_por_dominio"][dominio] = {
                "total_prompts": len(df_dominio),
                "tempo_medio": df_dominio["time"].mean(),
                "modelos_mais_rapidos": df_dominio.groupby("model")["time"].mean().nsmallest(3).to_dict()
            }
    
    # Ranking dos modelos por métricas
    if metricas:
        # Ranking por BLEU
        bleu_scores = {model: metrica['bleu']['bleu'] for model, metrica in metricas.items()}
        ranking_bleu = sorted(bleu_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Ranking por ROUGE
        rouge_scores = {model: metrica['rouge']['rouge1'] for model, metrica in metricas.items()}
        ranking_rouge = sorted(rouge_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Ranking por BERTScore
        bert_scores = {model: sum(metrica['bertscore']['f1'])/len(metrica['bertscore']['f1']) for model, metrica in metricas.items()}
        ranking_bert = sorted(bert_scores.items(), key=lambda x: x[1], reverse=True)
        
        report["ranking_modelos"] = {
            "por_bleu": ranking_bleu,
            "por_rouge": ranking_rouge,
            "por_bertscore": ranking_bert,
            "ranking_geral": {}
        }
        
        # Ranking geral (média das 3 métricas)
        for model in metricas.keys():
            bleu_norm = bleu_scores[model] / max(bleu_scores.values()) if max(bleu_scores.values()) > 0 else 0
            rouge_norm = rouge_scores[model] / max(rouge_scores.values()) if max(rouge_scores.values()) > 0 else 0
            bert_norm = bert_scores[model] / max(bert_scores.values()) if max(bert_scores.values()) > 0 else 0
            score_geral = (bleu_norm + rouge_norm + bert_norm) / 3
            report["ranking_modelos"]["ranking_geral"][model] = score_geral
        
        ranking_geral = sorted(report["ranking_modelos"]["ranking_geral"].items(), key=lambda x: x[1], reverse=True)
        report["ranking_modelos"]["ranking_geral"] = ranking_geral
    
    # Salva relatório em JSON
    report_path = os.path.join(folder_path, "relatorio_final.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Salva relatório em texto formatado para trabalho acadêmico
    txt_report_path = os.path.join(folder_path, "relatorio_final.txt")
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO FINAL - COMPARAÇÃO DE MODELOS DE LINGUAGEM\n")
        f.write("Análise Acadêmica para Trabalho de Faculdade\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. INFORMAÇÕES GERAIS DA EXECUÇÃO\n")
        f.write("-" * 50 + "\n")
        f.write(f"Data/Hora da Execução: {report['execucao']['data_hora']}\n")
        f.write(f"Tempo Total de Execução: {report['execucao']['tempo_execucao']:.2f} segundos\n")
        f.write(f"Total de Respostas Geradas: {report['execucao']['total_respostas']}\n")
        f.write(f"Modelos Testados: {', '.join(report['execucao']['modelos_testados'])}\n")
        f.write(f"Prompts Executados: {report['execucao']['prompts_executados']}\n\n")
        
        f.write("2. DISTRIBUIÇÃO DOS PROMPTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Prompts em Português: {report['execucao']['prompts_por_idioma']['portugues']}\n")
        f.write(f"Prompts em Inglês: {report['execucao']['prompts_por_idioma']['ingles']}\n")
        f.write(f"Prompts de Tecnologia (PT): {report['execucao']['prompts_por_dominio']['tecnologia_pt']}\n")
        f.write(f"Prompts de Ciência (PT): {report['execucao']['prompts_por_dominio']['ciencia_pt']}\n")
        f.write(f"Prompts de Business (EN): {report['execucao']['prompts_por_dominio']['business_en']}\n")
        f.write(f"Prompts de Tecnologia (EN): {report['execucao']['prompts_por_dominio']['tecnologia_en']}\n\n")
        
        f.write("3. ESTATÍSTICAS DE TEMPO\n")
        f.write("-" * 50 + "\n")
        for modelo, tempo in report['estatisticas_gerais']['tempo_medio_por_modelo'].items():
            desvio = report['estatisticas_gerais']['desvio_padrao_tempo'].get(modelo, 0)
            f.write(f"{modelo}: {tempo:.2f}s ± {desvio:.2f}s (médio ± desvio padrão)\n")
        
        f.write(f"\nModelo Mais Rápido: {report['estatisticas_gerais']['modelo_mais_rapido']}\n")
        f.write(f"Modelo Mais Lento: {report['estatisticas_gerais']['modelo_mais_lento']}\n\n")
        
        f.write("4. ANÁLISE DETALHADA DAS MÉTRICAS\n")
        f.write("-" * 50 + "\n")
        if metricas:
            f.write("4.1 MÉTRICAS POR MODELO:\n")
            for modelo, metrica in report['metricas_por_modelo'].items():
                f.write(f"\n{modelo.upper()}:\n")
                f.write(f"  BLEU Score: {metrica['bleu']['bleu']:.4f}\n")
                f.write(f"  ROUGE-1: {metrica['rouge']['rouge1']:.4f}\n")
                f.write(f"  ROUGE-2: {metrica['rouge']['rouge2']:.4f}\n")
                f.write(f"  ROUGE-L: {metrica['rouge']['rougeL']:.4f}\n")
                f.write(f"  BERTScore (F1): {sum(metrica['bertscore']['f1'])/len(metrica['bertscore']['f1']):.4f}\n")
            
            f.write("\n4.2 RANKING DOS MODELOS:\n")
            f.write("Por BLEU Score:\n")
            for i, (modelo, score) in enumerate(report['ranking_modelos']['por_bleu'], 1):
                f.write(f"  {i}º: {modelo} ({score:.4f})\n")
            
            f.write("\nPor ROUGE-1:\n")
            for i, (modelo, score) in enumerate(report['ranking_modelos']['por_rouge'], 1):
                f.write(f"  {i}º: {modelo} ({score:.4f})\n")
            
            f.write("\nPor BERTScore:\n")
            for i, (modelo, score) in enumerate(report['ranking_modelos']['por_bertscore'], 1):
                f.write(f"  {i}º: {modelo} ({score:.4f})\n")
            
            f.write("\nRanking Geral (média normalizada):\n")
            for i, (modelo, score) in enumerate(report['ranking_modelos']['ranking_geral'], 1):
                f.write(f"  {i}º: {modelo} ({score:.4f})\n")
        
        f.write("\n5. ANÁLISE POR DOMÍNIO\n")
        f.write("-" * 50 + "\n")
        for dominio, dados in report['analise_por_dominio'].items():
            f.write(f"\n{dominio}:\n")
            f.write(f"  Total de Prompts: {dados['total_prompts']}\n")
            f.write(f"  Tempo Médio: {dados['tempo_medio']:.2f}s\n")
            f.write(f"  Top 3 Modelos Mais Rápidos:\n")
            for modelo, tempo in dados['modelos_mais_rapidos'].items():
                f.write(f"    - {modelo}: {tempo:.2f}s\n")
        
        f.write(f"\n6. INFORMAÇÕES TÉCNICAS\n")
        f.write("-" * 50 + "\n")
        f.write(f"EvidentlyAI Disponível: {'Sim' if report['evidently_disponivel'] else 'Não'}\n")
        f.write(f"Arquivos Gerados: {', '.join(report['arquivos_gerados'])}\n")
        
        f.write(f"\n7. CONCLUSÕES E RECOMENDAÇÕES\n")
        f.write("-" * 50 + "\n")
        if metricas:
            melhor_modelo = report['ranking_modelos']['ranking_geral'][0][0] if report['ranking_modelos']['ranking_geral'] else "N/A"
            f.write(f"Modelo com Melhor Performance Geral: {melhor_modelo}\n")
            f.write(f"Modelo Mais Rápido: {report['estatisticas_gerais']['modelo_mais_rapido']}\n")
            f.write(f"Modelo Mais Lento: {report['estatisticas_gerais']['modelo_mais_lento']}\n\n")
            
            f.write("Recomendações:\n")
            f.write("- Para aplicações que priorizam qualidade: {}\n".format(report['ranking_modelos']['por_bleu'][0][0]))
            f.write("- Para aplicações que priorizam velocidade: {}\n".format(report['estatisticas_gerais']['modelo_mais_rapido']))
            f.write("- Para aplicações em português: Analisar performance específica nos prompts PT\n")
            f.write("- Para aplicações em inglês: Analisar performance específica nos prompts EN\n")
    
    return report_path, txt_report_path

def load_prompts():
    """
    Carrega prompts e referências do arquivo JSON.
    Retorna lista de prompts e lista de referências.
    """
    try:
        # Carregar prompts do arquivo JSON
        prompts_file = os.path.join(config.PROMPTS_FOLDER, config.PROMPTS_FILE)
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        
        # Verificar se é o novo formato estruturado (v2.0)
        if prompts_data.get('version') == '2.0':
            prompts_list = prompts_data.get('prompts', [])
            references = prompts_data.get('references', [])
            
            # Extrair apenas o texto do prompt do formato estruturado
            prompts = []
            for prompt_item in prompts_list:
                if isinstance(prompt_item, dict):
                    prompts.append(prompt_item.get('prompt', ''))
                else:
                    prompts.append(prompt_item)
        else:
            # Formato antigo (v1.0)
            prompts = prompts_data.get('prompts', [])
            references = prompts_data.get('references', [])
        
        # Garantir que temos o mesmo número de prompts e referências
        if len(references) != len(prompts):
            # Se não temos referências suficientes, preencher com strings vazias
            references = references + [""] * (len(prompts) - len(references))
        
        return prompts, references
    except FileNotFoundError:
        print(f"⚠️ Arquivo {config.PROMPTS_FILE} não encontrado, usando prompts padrão")
        # Fallback para prompts básicos
        prompts = [
            "O que é inteligência artificial?",
            "Explique machine learning.",
            "Como funciona deep learning?",
            "Quais são os desafios da IA?",
            "Descreva computação quântica."
        ]
        references = [""] * len(prompts)
        return prompts, references

def load_benchmark_prompts():
    """
    Carrega prompts de benchmarks padronizados (MMLU, HellaSwag) do arquivo JSON.
    Retorna lista de prompts, referências e informações de benchmark.
    """
    prompts = []
    references = []
    benchmark_info = []
    
    try:
        benchmarks_file = os.path.join(config.BENCHMARKS_FOLDER, config.BENCHMARKS_FILE)
        with open(benchmarks_file, 'r', encoding='utf-8') as f:
            benchmarks_data = json.load(f)
        
        for benchmark_name, benchmark_data in benchmarks_data['benchmarks'].items():
            if benchmark_name == 'mmlu':
                for subject, questions in benchmark_data['subjects'].items():
                    for q in questions:
                        prompt = format_mmlu_prompt(q)
                        prompts.append(prompt)
                        references.append(q['answer'])
                        benchmark_info.append({
                            'benchmark': 'mmlu',
                            'subject': subject,
                            'question_id': q['id']
                        })
            
            elif benchmark_name == 'hellaswag':
                for q in benchmark_data['questions']:
                    prompt = format_hellaswag_prompt(q)
                    prompts.append(prompt)
                    references.append(q['answer'])
                    benchmark_info.append({
                        'benchmark': 'hellaswag',
                        'question_id': q['id']
                    })
    
    except FileNotFoundError:
        print(f"⚠️ Arquivo {config.BENCHMARKS_FILE} não encontrado")
    
    return prompts, references, benchmark_info

def format_mmlu_prompt(question_data):
    """
    Formata prompt para MMLU (Massive Multitask Language Understanding).
    """
    choices = "\n".join([f"{choice}" for choice in question_data['choices']])
    return f"Question: {question_data['question']}\n\nChoices:\n{choices}\n\nAnswer:"

def format_hellaswag_prompt(question_data):
    """
    Formata prompt para HellaSwag (Commonsense Reasoning).
    """
    choices = "\n".join([f"{choice}" for choice in question_data['choices']])
    return f"Context: {question_data['context']}\n\nQuestion: {question_data['question']}\n\nChoices:\n{choices}\n\nAnswer:" 