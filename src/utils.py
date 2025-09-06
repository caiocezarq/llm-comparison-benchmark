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
    Retorna lista de prompts e referências/gabaritos para avaliação dos modelos.
    Prompts diversificados em português e inglês para análise abrangente.
    """
    prompts = [
        # Prompts em Português - Tecnologia e IA (melhorados)
        "O que é inteligência artificial e como ela está transformando o mundo? Forneça uma explicação detalhada com exemplos práticos.",
        "Explique a diferença entre machine learning e deep learning, incluindo casos de uso específicos para cada abordagem.",
        "Como funcionam os modelos de linguagem como GPT e LLaMA? Descreva a arquitetura e o processo de treinamento.",
        "Quais são os principais desafios éticos da IA? Discuta implicações sociais e possíveis soluções.",
        "Descreva o conceito de computação quântica e suas aplicações práticas atuais e futuras.",
        
        # Prompts em Português - Ciência e Meio Ambiente (melhorados)
        "Explique o impacto das mudanças climáticas na biodiversidade, incluindo exemplos específicos de espécies afetadas.",
        "Como funciona a fotossíntese e por que é importante para a vida na Terra? Inclua detalhes sobre o processo químico.",
        "Descreva o processo de evolução das espécies, explicando os mecanismos de seleção natural e adaptação.",
        "Quais são as principais fontes de energia renovável? Compare vantagens e desvantagens de cada uma.",
        "Explique o conceito de sustentabilidade corporativa e como as empresas podem implementá-la efetivamente.",
        
        # Prompts em Inglês - Business e Inovação (melhorados)
        "What are the key factors for successful digital transformation? Provide a comprehensive framework with real-world examples.",
        "Explain the concept of disruptive innovation in business, including historical examples and current trends.",
        "How does artificial intelligence impact customer experience? Discuss specific applications and measurable benefits.",
        "What are the main challenges of remote work management? Provide strategies and tools for effective remote leadership.",
        "Describe the future of e-commerce and online retail, including emerging technologies and consumer behavior trends.",
        
        # Prompts em Inglês - Ciência e Tecnologia (melhorados)
        "Explain the principles of quantum computing and its potential applications in cryptography, optimization, and simulation.",
        "How do neural networks process and learn from data? Describe the mathematical foundations and learning algorithms.",
        "What are the latest developments in renewable energy technology? Focus on breakthrough innovations and scalability.",
        "Describe the impact of 5G technology on modern communication, including benefits for IoT and smart cities.",
        "How does blockchain technology ensure data security and transparency? Explain consensus mechanisms and use cases."
    ]
    
    references = [
        # Referências em Português - Tecnologia e IA (melhoradas e mais longas)
        "A inteligência artificial (IA) é um campo interdisciplinar da ciência da computação que visa criar sistemas capazes de realizar tarefas que tradicionalmente exigiriam inteligência humana. Estes sistemas utilizam algoritmos de aprendizado de máquina, processamento de linguagem natural e visão computacional para analisar dados, reconhecer padrões e tomar decisões autônomas. A IA está transformando diversos setores: na saúde, auxilia no diagnóstico médico e descoberta de medicamentos; na indústria, otimiza processos de produção e controle de qualidade; no transporte, desenvolve veículos autônomos; e na educação, personaliza o aprendizado. Exemplos práticos incluem assistentes virtuais como Siri e Alexa, sistemas de recomendação da Netflix e Amazon, carros autônomos da Tesla, e ferramentas de tradução automática. A IA também está revolucionando a pesquisa científica, acelerando descobertas em áreas como genética, física e química através da análise de grandes volumes de dados.",
        
        "Machine learning (ML) é um subconjunto da inteligência artificial que permite aos sistemas aprenderem e melhorarem automaticamente através da experiência, sem serem explicitamente programados para cada tarefa. Utiliza algoritmos estatísticos para identificar padrões em dados e fazer previsões ou decisões. Deep learning, por sua vez, é uma subcategoria do machine learning que emprega redes neurais artificiais com múltiplas camadas (daí o termo 'deep') para processar dados de forma hierárquica. Enquanto o ML tradicional funciona bem com dados estruturados e características pré-definidas, o deep learning pode processar dados não estruturados como imagens, texto e áudio, extraindo características automaticamente. Casos de uso do ML incluem sistemas de recomendação, detecção de fraudes, análise de sentimentos e previsão de demanda. O deep learning é especialmente eficaz em reconhecimento de imagens (como em diagnósticos médicos), processamento de linguagem natural (tradução, chatbots), e sistemas de reconhecimento de voz.",
        
        "Os modelos de linguagem como GPT (Generative Pre-trained Transformer) e LLaMA (Large Language Model Meta AI) são baseados na arquitetura Transformer, introduzida em 2017. Esta arquitetura utiliza mecanismos de atenção para processar sequências de texto, permitindo que o modelo compreenda relações contextuais entre palavras distantes. O processo de treinamento ocorre em duas fases: pré-treinamento e fine-tuning. Na fase de pré-treinamento, o modelo é exposto a vastas quantidades de texto da internet para aprender padrões linguísticos, gramática e conhecimento factual. Na fase de fine-tuning, o modelo é ajustado para tarefas específicas usando datasets menores e mais focados. O GPT utiliza uma abordagem de decodificação autoregressiva, gerando texto token por token, enquanto o LLaMA emprega uma arquitetura similar mas com otimizações específicas. Estes modelos são capazes de gerar texto coerente, responder perguntas, traduzir idiomas e realizar raciocínio complexo, representando um avanço significativo na capacidade de processamento de linguagem natural.",
        
        "Os principais desafios éticos da IA incluem múltiplas dimensões que requerem atenção cuidadosa. O viés algorítmico surge quando modelos de IA perpetuam ou amplificam preconceitos presentes nos dados de treinamento, resultando em discriminação contra grupos minoritários em áreas como contratação, empréstimos e justiça criminal. A privacidade de dados representa outro desafio crítico, pois sistemas de IA frequentemente requerem acesso a informações pessoais sensíveis, criando riscos de vazamento ou uso indevido. A automação de empregos gera preocupações sobre desemprego em massa, especialmente em setores que dependem de tarefas repetitivas. A transparência e explicabilidade dos sistemas de IA são essenciais para garantir que decisões automatizadas possam ser compreendidas e contestadas. Questões de responsabilidade legal surgem quando sistemas de IA causam danos, criando incertezas sobre quem deve ser responsabilizado. A segurança cibernética representa outro desafio, pois sistemas de IA podem ser vulneráveis a ataques adversariais. Soluções incluem desenvolvimento de frameworks éticos, regulamentação governamental, auditoria de algoritmos e investimento em educação para preparar a força de trabalho para a era da IA.",
        
        "A computação quântica representa uma revolução paradigmática na forma como processamos informações, utilizando princípios da mecânica quântica como superposição, emaranhamento e interferência quântica. Diferentemente dos bits clássicos que existem em estados binários (0 ou 1), os qubits quânticos podem existir em superposição, permitindo processamento paralelo massivo. Aplicações práticas atuais incluem simulação molecular para descoberta de medicamentos, otimização de portfólios financeiros, e criptografia quântica para comunicações seguras. No futuro, a computação quântica promete revolucionar áreas como inteligência artificial, modelagem climática, e resolução de problemas de otimização complexos. Empresas como IBM, Google e Microsoft já oferecem acesso a computadores quânticos através de serviços em nuvem. No entanto, desafios significativos permanecem, incluindo a necessidade de temperaturas extremamente baixas, correção de erros quânticos, e escalabilidade. A computação quântica não substituirá completamente a computação clássica, mas complementará em problemas específicos onde sua vantagem quântica seja significativa.",
        
        # Referências em Português - Ciência e Meio Ambiente
        "As mudanças climáticas afetam habitats naturais, causando extinção de espécies, alterações nos padrões migratórios e desequilíbrios nos ecossistemas terrestres e marinhos.",
        "A fotossíntese é o processo pelo qual plantas convertem luz solar, CO2 e água em oxigênio e glicose, sendo fundamental para a vida na Terra e o ciclo do carbono.",
        "A evolução das espécies ocorre através de seleção natural, mutações genéticas e adaptação ao ambiente, resultando na diversidade biológica observada.",
        "As principais fontes incluem energia solar, eólica, hidrelétrica, biomassa e geotérmica, oferecendo alternativas sustentáveis aos combustíveis fósseis.",
        "A sustentabilidade corporativa integra práticas ambientais, sociais e de governança (ESG) nas operações empresariais para criar valor a longo prazo.",
        
        # Referências em Inglês - Business e Inovação
        "Key factors include strong leadership commitment, employee training, technology infrastructure, change management processes, and continuous improvement culture.",
        "Disruptive innovation creates new markets by offering simpler, more affordable alternatives that eventually displace established products and services.",
        "AI enhances customer experience through personalization, predictive analytics, chatbots, and automated customer service, improving satisfaction and loyalty.",
        "Main challenges include communication barriers, team collaboration, work-life balance, technology infrastructure, and maintaining company culture remotely.",
        "The future includes AI-powered personalization, augmented reality shopping, voice commerce, social commerce, and seamless omnichannel experiences.",
        
        # Referências em Inglês - Ciência e Tecnologia
        "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information, potentially solving complex problems in cryptography and optimization.",
        "Neural networks process data through interconnected nodes that learn patterns by adjusting weights based on training data and error feedback.",
        "Latest developments include advanced solar panels, offshore wind farms, hydrogen fuel cells, and energy storage solutions like next-generation batteries.",
        "5G technology provides faster data transmission, lower latency, and greater connectivity, enabling IoT, autonomous vehicles, and smart city applications.",
        "Blockchain ensures security through cryptographic hashing, distributed ledger technology, and consensus mechanisms that prevent tampering and ensure transparency."
    ]
    
    return prompts, references 