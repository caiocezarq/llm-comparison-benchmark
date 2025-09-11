#!/usr/bin/env python3
"""
Pipeline de Comparação de Modelos de Linguagem
Script principal para executar o pipeline completo.
"""
import sys
import os
import time
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Adicionar o diretório atual ao path para importar o módulo src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline, evaluate_and_export, generate_final_report
from src.utils import get_next_result_folder
from src.config import get_config, ConfigValidator
from src.logger import get_logger, log_execution_start, log_execution_end, log_configuration, log_statistics

# =============================================================================
# CONFIGURAÇÕES DE EXECUÇÃO
# =============================================================================
# Carregar configurações centralizadas
config = get_config()

# Validar configurações
try:
    ConfigValidator.validate_all()
    print("✅ Configurações validadas com sucesso")
except ValueError as e:
    print(f"❌ Erro na configuração: {e}")
    sys.exit(1)

# Configurar logger
logger = get_logger("main")
# =============================================================================

def main():
    """
    Função principal para executar o pipeline múltiplas vezes.
    """
    print("🚀 PIPELINE DE COMPARAÇÃO DE MODELOS DE LINGUAGEM")
    print("=" * 60)
    print("📊 Modelos suportados: Groq + Google Gemini")
    print("🎯 Métricas: BLEU, ROUGE, BERTScore, EvidentlyAI")
    if config.INCLUDE_BENCHMARKS:
        print("🏆 Benchmarks: MMLU, HellaSwag")
    print("=" * 60)
    print(f"🔄 Execuções configuradas: {config.NUMERO_EXECUCOES}")
    print(f"⏱️  Timeout entre execuções: {config.TIMEOUT_ENTRE_EXECUCOES}s")
    if config.INCLUDE_BENCHMARKS:
        print(f"🏆 Benchmarks incluídos: {config.INCLUDE_BENCHMARKS}")
    print("=" * 60)
    
    # Log das configurações
    log_configuration(logger, {
        "NUMERO_EXECUCOES": config.NUMERO_EXECUCOES,
        "TIMEOUT_ENTRE_EXECUCOES": config.TIMEOUT_ENTRE_EXECUCOES,
        "TIMEOUT_ENTRE_PERGUNTAS": config.TIMEOUT_ENTRE_PERGUNTAS,
        "PASTA_RESULTADOS": config.PASTA_RESULTADOS,
        "PREFIXO_EXECUCAO": config.PREFIXO_EXECUCAO
    })
    
    execucoes_sucesso = 0
    execucoes_erro = 0
    tempo_inicio_total = time.time()
    
    for execucao in range(1, config.NUMERO_EXECUCOES + 1):
        print(f"\n🔄 EXECUÇÃO {execucao}/{config.NUMERO_EXECUCOES}")
        print("-" * 40)
        
        # Log do início da execução
        log_execution_start(logger, execucao, config.NUMERO_EXECUCOES)
        
        try:
            tempo_inicio = time.time()
            
            # Criar pasta individual para esta execução
            result_folder = get_next_result_folder()
            print(f"📁 Resultados da execução {execucao} serão salvos em: {result_folder}")
            
            # Executar pipeline
            df = run_pipeline(include_benchmarks=config.INCLUDE_BENCHMARKS)
            
            tempo_execucao = time.time() - tempo_inicio
            
            if df is not None and not df.empty:
                execucoes_sucesso += 1
                print(f"✅ Execução {execucao} concluída com sucesso!")
                print(f"📊 Total de resultados: {len(df)}")
                print(f"🤖 Modelos testados: {df['model'].nunique()}")
                print(f"📝 Prompts executados: {df['prompt'].nunique()}")
                print(f"⏱️  Tempo de execução: {tempo_execucao:.2f}s")
                
                # Mostrar estatísticas básicas
                print(f"📈 Tempo médio por resposta: {df['time'].mean():.2f}s")
                print(f"📏 Comprimento médio das respostas: {df['prediction'].str.len().mean():.0f} caracteres")
                
                # Mostrar modelos com sucesso
                success_models = df[~df['is_error']]['model'].unique()
                print(f"✅ Modelos funcionando: {len(success_models)}")
                
                # Mostrar modelos com erro
                error_models = df[df['is_error']]['model'].unique()
                error_count = df['is_error'].sum()
                if len(error_models) > 0:
                    print(f"❌ Modelos com erro: {len(error_models)}")
                    print(f"📊 Total de erros: {error_count}/{len(df)} ({(error_count/len(df)*100):.1f}%)")
                
                # Salvar resultados da execução
                print(f"💾 Salvando resultados da execução {execucao}...")
                try:
                    # Exporta resultados das APIs (sem cálculos de métricas)
                    stats = evaluate_and_export(df, result_folder)
                    
                    # Gera relatório básico da pipeline
                    report_json, report_txt = generate_final_report(df, stats, {}, tempo_execucao, result_folder)
                    
                    print(f"✅ Resultados salvos com sucesso!")
                    
                except Exception as e:
                    print(f"⚠️  Erro ao salvar resultados: {e}")
                
                # Log do fim da execução (sucesso)
                log_execution_end(logger, execucao, True, tempo_execucao)
                
            else:
                execucoes_erro += 1
                print(f"❌ Execução {execucao} falhou: Nenhum resultado foi gerado")
                
                # Log do fim da execução (erro)
                log_execution_end(logger, execucao, False, tempo_execucao)
                
        except Exception as e:
            execucoes_erro += 1
            print(f"❌ Erro na execução {execucao}: {e}")
            
            # Log do fim da execução (erro)
            log_execution_end(logger, execucao, False, time.time() - tempo_inicio)
        
        # Aguardar entre execuções (exceto na última)
        if execucao < config.NUMERO_EXECUCOES:
            print(f"\n⏳ Aguardando {config.TIMEOUT_ENTRE_EXECUCOES}s antes da próxima execução...")
            time.sleep(config.TIMEOUT_ENTRE_EXECUCOES)
    
    # Resumo final
    tempo_total = time.time() - tempo_inicio_total
    print("\n" + "=" * 60)
    print("📋 RESUMO FINAL DAS EXECUÇÕES")
    print("=" * 60)
    print(f"✅ Execuções bem-sucedidas: {execucoes_sucesso}/{config.NUMERO_EXECUCOES}")
    print(f"❌ Execuções com erro: {execucoes_erro}/{config.NUMERO_EXECUCOES}")
    print(f"⏱️  Tempo total: {tempo_total:.2f}s ({tempo_total/60:.1f} minutos)")
    print(f"📈 Taxa de sucesso: {(execucoes_sucesso/config.NUMERO_EXECUCOES)*100:.1f}%")
    
    # Estatísticas de erros se houver execuções bem-sucedidas
    if execucoes_sucesso > 0:
        print(f"\n📊 ANÁLISE DE ERROS:")
        print(f"💡 Verifique os arquivos 'relatorio_erros.json' e 'relatorio_erros.txt' para detalhes")
        print(f"📁 Pasta de resultados: {result_folder}")
    
    # Log das estatísticas finais
    log_statistics(logger, {
        "execucoes_sucesso": execucoes_sucesso,
        "execucoes_erro": execucoes_erro,
        "tempo_total": f"{tempo_total:.2f}s",
        "taxa_sucesso": f"{(execucoes_sucesso/config.NUMERO_EXECUCOES)*100:.1f}%"
    })
    
    if execucoes_sucesso > 0:
        print(f"\n📁 Resultados salvos na pasta: {result_folder}")
        # Executar análise consolidada se houver execuções bem-sucedidas
        if execucoes_sucesso > 0:
            print("\n🔬 Iniciando análise consolidada...")
            try:
                from analysis.analysis import executar_analise
                resultado_analise = executar_analise()
                if resultado_analise:
                    print("✅ Análise consolidada concluída com sucesso!")
                else:
                    print("⚠️ Análise consolidada não pôde ser executada")
            except Exception as e:
                print(f"❌ Erro na análise consolidada: {e}")
        else:
            print("💡 Execute mais execuções para permitir análise consolidada")
        
        print(f"🎯 Análise executada com {execucoes_sucesso} execuções bem-sucedidas")
    
    return 0 if execucoes_sucesso > 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
