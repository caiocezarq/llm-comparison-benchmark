# pipeline.py
"""
Pipeline principal para rodar prompts em múltiplos modelos, avaliar e exportar resultados.
"""
import pandas as pd
import time
import os
import warnings
import logging
import json
from datetime import datetime

# Suprimir avisos desnecessários
warnings.filterwarnings("ignore")
logging.getLogger("groq").setLevel(logging.ERROR)

from .models import ModelRunner, AVAILABLE_MODELS
from .utils import save_results_csv, save_results_json, load_prompts, load_benchmark_prompts, get_next_result_folder, generate_final_report
from .config import get_config

# Carregar configurações
config = get_config()


def run_pipeline(api_key=None, model_keys=None, include_benchmarks=False):
    """
    Executa prompts em todos os modelos especificados e retorna DataFrame com resultados.
    
    Args:
        api_key: Chave da API (opcional)
        model_keys: Lista de modelos para testar (opcional)
        include_benchmarks: Se True, inclui prompts de benchmarks padronizados
    """
    # Carregar prompts padrão
    prompts, references = load_prompts()
    benchmark_info = []
    
    # Adicionar prompts de benchmarks se solicitado
    if include_benchmarks:
        benchmark_prompts, benchmark_refs, benchmark_info = load_benchmark_prompts()
        prompts = prompts + benchmark_prompts
        references = references + benchmark_refs
    
    all_results = []
    if model_keys is None:
        model_keys = list(AVAILABLE_MODELS.keys())
    
    for model_key in model_keys:
        model_id = AVAILABLE_MODELS[model_key]
        print(f"Executando modelo: {model_key} ({model_id})")
        
        try:
            runner = ModelRunner(model_key, api_key=api_key)
        except Exception as e:
            print(f"❌ Erro ao inicializar modelo {model_key}: {e}")
            continue
            
        for i, (prompt, reference) in enumerate(zip(prompts, references)):
            print(f"  Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            start = time.time()
            
            try:
                prediction = runner.generate(prompt)
                # Garantir que a predição seja string válida
                if isinstance(prediction, str):
                    prediction = prediction.encode('utf-8', errors='ignore').decode('utf-8')
                else:
                    prediction = str(prediction)
                
                # Verificar se é um erro
                if prediction.startswith("[ERRO]"):
                    print(f"    ⚠️  Erro: {prediction}")
                else:
                    print(f"    ✅ Resposta gerada ({len(prediction)} chars)")
                    
            except Exception as e:
                prediction = f"[ERRO]: Exceção não tratada - {str(e)}"
                print(f"    ❌ Exceção: {e}")
            
            elapsed = time.time() - start
            # Adicionar informações de benchmark se disponível
            result = {
                "model": model_key,
                "prompt": prompt,
                "reference": reference,
                "prediction": prediction,
                "time": elapsed,
                "timestamp": datetime.now().isoformat(),
                "prompt_length": len(prompt),
                "response_length": len(prediction),
                "is_error": prediction.startswith('[ERRO]')
            }
            
            # Adicionar informações de benchmark se disponível
            if i < len(benchmark_info):
                result.update(benchmark_info[i])
            else:
                result['benchmark'] = None
                result['subject'] = None
                result['question_id'] = None
            
            all_results.append(result)
            
            # Timeout entre perguntas (exceto na última pergunta do modelo)
            if i < len(prompts) - 1 and config.TIMEOUT_ENTRE_PERGUNTAS > 0:
                print(f"    ⏳ Aguardando {config.TIMEOUT_ENTRE_PERGUNTAS}s antes da próxima pergunta...")
                time.sleep(config.TIMEOUT_ENTRE_PERGUNTAS)
    
    # Criar DataFrame com encoding correto
    df = pd.DataFrame(all_results)
    
    # Garantir que todas as colunas de texto tenham encoding correto
    text_columns = ['prompt', 'reference', 'prediction']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: x.encode('utf-8', errors='ignore').decode('utf-8')
            )
    
    # Estatísticas de erros
    error_count = df['is_error'].sum()
    total_count = len(df)
    print(f"\n📊 Estatísticas: {error_count}/{total_count} erros encontrados")
    
    # Análise detalhada de erros
    if error_count > 0:
        print(f"\n🔍 Análise de erros:")
        error_analysis = analyze_errors(df)
        for model, errors in error_analysis.items():
            print(f"  {model}: {len(errors)} erros")
            for error_type, count in errors.items():
                print(f"    - {error_type}: {count}")
    
    return df


def analyze_errors(df):
    """
    Analisa erros nos resultados e retorna estatísticas por modelo.
    
    Args:
        df (pd.DataFrame): DataFrame com os resultados
        
    Returns:
        dict: Análise de erros por modelo
    """
    error_analysis = {}
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        errors = model_data[model_data['is_error']]
        
        if len(errors) > 0:
            error_types = {}
            for error in errors['prediction']:
                # Extrair tipo de erro
                if 'Rate limit' in error:
                    error_type = 'Rate Limit'
                elif 'Timeout' in error:
                    error_type = 'Timeout'
                elif 'Authentication' in error:
                    error_type = 'Authentication'
                elif 'Not found' in error:
                    error_type = 'Model Not Found'
                elif 'Safety' in error or 'blocked' in error:
                    error_type = 'Content Blocked'
                elif 'Context length' in error or 'token' in error:
                    error_type = 'Context Length'
                elif 'Permission' in error:
                    error_type = 'Permission Denied'
                else:
                    error_type = 'Other Error'
                
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            error_analysis[model] = error_types
    
    return error_analysis


def evaluate_and_export(df, folder_path):
    """
    Exporta resultados das APIs em CSV/JSON na pasta especificada.
    A pipeline apenas coleta respostas das APIs - métricas são calculadas na análise detalhada.
    """
    print("💾 Exportando resultados das APIs...")
    
    # Salva arquivos por modelo
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        csv_path = os.path.join(folder_path, f"resultados_{model}.csv")
        save_results_csv(sub, csv_path)
        print(f"  ✅ {model}: {len(sub)} respostas salvas")
    
    # Salva arquivos consolidados
    csv_path = os.path.join(folder_path, "resultados_todos.csv")
    json_path = os.path.join(folder_path, "resultados_todos.json")
    save_results_csv(df, csv_path)
    save_results_json(df, json_path)
    print(f"  ✅ Dados consolidados: {len(df)} registros")
    
    # Salva relatório de erros se houver
    error_count = df['is_error'].sum()
    if error_count > 0:
        print("📋 Gerando relatório de erros...")
        save_error_report(df, folder_path)
    
    # Estatísticas básicas
    stats = {
        "total_respostas": len(df),
        "modelos_unicos": df['model'].nunique(),
        "respostas_validas": len(df[~df['is_error']]),
        "respostas_com_erro": error_count,
        "taxa_erro": (error_count / len(df)) * 100 if len(df) > 0 else 0
    }
    
    print(f"📊 Estatísticas: {stats['respostas_validas']}/{stats['total_respostas']} respostas válidas ({100-stats['taxa_erro']:.1f}%)")
    
    return stats


def save_error_report(df, folder_path):
    """
    Salva relatório detalhado de erros.
    
    Args:
        df (pd.DataFrame): DataFrame com os resultados
        folder_path (str): Pasta onde salvar o relatório
    """
    import json
    from datetime import datetime
    
    # Análise de erros
    error_analysis = analyze_errors(df)
    
    # Estatísticas gerais de erro
    total_errors = df['is_error'].sum()
    total_responses = len(df)
    error_rate = (total_errors / total_responses) * 100 if total_responses > 0 else 0
    
    # Relatório de erros
    error_report = {
        "resumo": {
            "total_erros": int(total_errors),
            "total_respostas": int(total_responses),
            "taxa_erro": round(error_rate, 2),
            "data_analise": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "erros_por_modelo": error_analysis,
        "detalhes_erros": []
    }
    
    # Detalhes dos erros
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        errors = model_data[model_data['is_error']]
        
        for _, row in errors.iterrows():
            error_detail = {
                "modelo": model,
                "prompt": row['prompt'][:100] + "..." if len(row['prompt']) > 100 else row['prompt'],
                "erro": row['prediction'],
                "tempo": row['time']
            }
            error_report["detalhes_erros"].append(error_detail)
    
    # Salva relatório JSON
    error_report_path = os.path.join(folder_path, "relatorio_erros.json")
    with open(error_report_path, 'w', encoding='utf-8') as f:
        json.dump(error_report, f, indent=2, ensure_ascii=False)
    
    # Salva relatório TXT
    error_txt_path = os.path.join(folder_path, "relatorio_erros.txt")
    with open(error_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO DE ERROS - PIPELINE DE COMPARAÇÃO DE MODELOS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Data da Análise: {error_report['resumo']['data_analise']}\n")
        f.write(f"Total de Erros: {error_report['resumo']['total_erros']}\n")
        f.write(f"Total de Respostas: {error_report['resumo']['total_respostas']}\n")
        f.write(f"Taxa de Erro: {error_report['resumo']['taxa_erro']}%\n\n")
        
        f.write("ERROS POR MODELO:\n")
        f.write("-" * 50 + "\n")
        for model, errors in error_analysis.items():
            f.write(f"\n{model}:\n")
            for error_type, count in errors.items():
                f.write(f"  - {error_type}: {count}\n")
        
        f.write("\nDETALHES DOS ERROS:\n")
        f.write("-" * 50 + "\n")
        for detail in error_report["detalhes_erros"]:
            f.write(f"\nModelo: {detail['modelo']}\n")
            f.write(f"Prompt: {detail['prompt']}\n")
            f.write(f"Erro: {detail['erro']}\n")
            f.write(f"Tempo: {detail['tempo']:.2f}s\n")
            f.write("-" * 30 + "\n")
    
    print(f"✅ Relatório de erros salvo: {error_report_path}")
    print(f"✅ Relatório de erros (TXT): {error_txt_path}")




def generate_final_report(df, stats, relatorios, tempo_execucao, folder_path):
    """
    Gera relatório básico da pipeline (apenas coleta de APIs).
    
    Args:
        df (pd.DataFrame): DataFrame com os resultados
        stats (dict): Estatísticas básicas
        relatorios (dict): Relatórios (vazio na pipeline)
        tempo_execucao (float): Tempo de execução em segundos
        folder_path (str): Pasta onde salvar o relatório
        
    Returns:
        tuple: (caminho_json, caminho_txt)
    """
    print("📝 Gerando relatório básico da pipeline...")
    
    # Preparar dados do relatório
    report_data = {
        "resumo": {
            "data_execucao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tempo_execucao": tempo_execucao,
            "total_respostas": len(df),
            "modelos_testados": df["model"].nunique(),
            "prompts_executados": df["prompt"].nunique(),
            "respostas_validas": stats.get("respostas_validas", 0),
            "taxa_erro": stats.get("taxa_erro", 0)
        },
        "modelos": {},
        "observacao": "Métricas acadêmicas (BLEU, ROUGE, BERTScore, EvidentlyAI) são calculadas na análise detalhada"
    }
    
    # Estatísticas por modelo
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        valid_responses = model_df[~model_df["is_error"]]
        
        report_data["modelos"][model] = {
            "total_respostas": len(model_df),
            "respostas_validas": len(valid_responses),
            "taxa_sucesso": (len(valid_responses) / len(model_df)) * 100 if len(model_df) > 0 else 0,
            "tempo_medio": model_df["time"].mean(),
            "comprimento_medio": valid_responses["prediction"].str.len().mean() if len(valid_responses) > 0 else 0
        }
    
    # Salvar relatório JSON
    json_path = os.path.join(folder_path, "relatorio_pipeline.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    # Salvar relatório TXT
    txt_path = os.path.join(folder_path, "relatorio_pipeline.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO DA PIPELINE - COLETA DE RESPOSTAS DAS APIs\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Data da Execução: {report_data['resumo']['data_execucao']}\n")
        f.write(f"Tempo de Execução: {report_data['resumo']['tempo_execucao']:.2f}s\n")
        f.write(f"Total de Respostas: {report_data['resumo']['total_respostas']}\n")
        f.write(f"Respostas Válidas: {report_data['resumo']['respostas_validas']}\n")
        f.write(f"Taxa de Erro: {report_data['resumo']['taxa_erro']:.1f}%\n")
        f.write(f"Modelos Testados: {report_data['resumo']['modelos_testados']}\n")
        f.write(f"Prompts Executados: {report_data['resumo']['prompts_executados']}\n\n")
        
        f.write("ESTATÍSTICAS POR MODELO:\n")
        f.write("-" * 50 + "\n")
        for model, stats in report_data["modelos"].items():
            f.write(f"\n{model}:\n")
            f.write(f"  Total de Respostas: {stats['total_respostas']}\n")
            f.write(f"  Respostas Válidas: {stats['respostas_validas']}\n")
            f.write(f"  Taxa de Sucesso: {stats['taxa_sucesso']:.1f}%\n")
            f.write(f"  Tempo Médio: {stats['tempo_medio']:.2f}s\n")
            f.write(f"  Comprimento Médio: {stats['comprimento_medio']:.0f} chars\n")
        
        f.write(f"\nOBSERVAÇÃO:\n")
        f.write(f"{report_data['observacao']}\n")
    
    print(f"✅ Relatório da pipeline salvo: {json_path}")
    print(f"✅ Relatório da pipeline (TXT): {txt_path}")
    
    return json_path, txt_path


def main():
    """
    Função principal para execução direta do script.
    """
    from dotenv import load_dotenv
    import os
    
    # Carrega variáveis de ambiente
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("Erro: API Key GROQ_API_KEY não encontrada no arquivo .env")
        return
    
    print("🚀 Iniciando pipeline de comparação de modelos...")
    
    # Cria pasta para resultados
    result_folder = get_next_result_folder()
    print(f"📁 Resultados serão salvos em: {result_folder}")
    
    
    # Executa pipeline
    start_time = time.time()
    df = run_pipeline(api_key=api_key)
    execution_time = time.time() - start_time
    
    # Avalia e exporta resultados
    metricas = evaluate_and_export(df, result_folder)
    
    # Gera relatório final consolidado
    report_json, report_txt = generate_final_report(df, metricas, {}, execution_time, result_folder)
    
    print("✅ Pipeline concluído!")
    print(f"📁 Pasta de resultados: {result_folder}")
    print(f"📊 Arquivos gerados:")
    print(f"   - resultados_todos.csv")
    print(f"   - resultados_todos.json")
    print(f"   - relatorio_final.json")
    print(f"   - relatorio_final.txt")
    for model in df["model"].unique():
        print(f"   - resultados_{model}.csv")
    
    print(f"⏱️  Tempo total de execução: {execution_time:.2f} segundos")
    print(f"📈 Métricas calculadas: {list(metricas.keys())}")
    
    print("📋 Relatórios EvidentlyAI: Serão calculados na análise detalhada")
    
    # Resumo das análises geradas
    print(f"\n🎓 ARQUIVOS PARA TRABALHO ACADÊMICO:")
    print(f"   📊 Análise comparativa: analise_comparativa.csv")
    print(f"   🏆 Ranking dos modelos: ranking_modelos.csv")
    print(f"   📋 Resultados por modelo: resultados_[modelo].csv")
    print(f"   📄 Relatório completo: relatorio_final.txt")
    print(f"   🔍 Dados JSON: relatorio_final.json")


if __name__ == "__main__":
    main() 