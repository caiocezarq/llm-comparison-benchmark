#!/usr/bin/env python3
"""
Teste Rápido do Sistema de Análise
Valida se o sistema está funcionando corretamente sem executar a pipeline completa.
"""

import sys
import os
from pathlib import Path

# Evita erro de encoding no terminal Windows (cp1252) ao imprimir emojis/acentos
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Adicionar o diretório atual ao path (já estamos no diretório principal)
sys.path.append(os.path.dirname(__file__))

def testar_imports():
    """Testa se todos os imports necessários funcionam."""
    print("🔍 Testando imports...")
    
    try:
        import pandas as pd
        print("   ✅ pandas")
    except ImportError as e:
        print(f"   ❌ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("   ✅ numpy")
    except ImportError as e:
        print(f"   ❌ numpy: {e}")
        return False
    
    try:
        from src.models import ModelRunner, AVAILABLE_MODELS
        print("   ✅ src.models")
    except ImportError as e:
        print(f"   ❌ src.models: {e}")
        return False
    
    try:
        from src.pipeline import run_pipeline
        print("   ✅ src.pipeline")
    except ImportError as e:
        print(f"   ❌ src.pipeline: {e}")
        return False
    
    return True

def testar_estrutura_arquivos():
    """Testa se a estrutura de arquivos está correta."""
    print("\n📁 Testando estrutura de arquivos...")
    
    arquivos_necessarios = [
        "main.py",
        "src/models.py",
        "src/pipeline.py",
        "src/utils.py",
        "src/config.py",
        "src/logger.py",
        "requirements.txt",
        "env_example.txt"
    ]
    
    for arquivo in arquivos_necessarios:
        if os.path.exists(arquivo):
            print(f"   ✅ {arquivo}")
        else:
            print(f"   ❌ {arquivo} - não encontrado")
            return False
    
    return True

def testar_pasta_resultados():
    """Testa se a pasta de resultados existe e tem execuções."""
    print("\n📊 Testando pasta de resultados...")
    
    pasta_results = "results"
    if not os.path.exists(pasta_results):
        print(f"   ⚠️ Pasta {pasta_results} não existe - será criada na primeira execução")
        return True
    
    # Contar execuções
    execucoes = [d for d in os.listdir(pasta_results) if d.startswith('resultado_')]
    print(f"   📈 Execuções encontradas: {len(execucoes)}")
    
    if len(execucoes) > 0:
        print(f"   📁 Execuções: {', '.join(sorted(execucoes))}")
        
        # Verificar se há execuções válidas
        execucoes_validas = 0
        for execucao in execucoes:
            arquivo_resultados = os.path.join(pasta_results, execucao, "resultados_todos.csv")
            if os.path.exists(arquivo_resultados):
                execucoes_validas += 1
        
        print(f"   ✅ Execuções válidas: {execucoes_validas}/{len(execucoes)}")
        
        if execucoes_validas >= 10:
            print("   🎯 Suficientes para análise de múltiplas execuções!")
        else:
            print(f"   💡 Execute mais {10 - execucoes_validas} vezes para análise completa")
    
    return True

def testar_analisador():
    """Testa se o sistema de análise funciona."""
    print("\n🔬 Testando sistema de análise...")
    
    try:
        from analysis.analysis import AnalysisSystem
        print("   ✅ Import do sistema de análise")
        
        # Criar instância
        sistema = AnalysisSystem()
        print("   ✅ Criação da instância")
        
        # Testar encontrar execuções
        execucoes = sistema.encontrar_execucoes()
        print(f"   ✅ Encontrou {len(execucoes)} execuções")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro no sistema de análise: {e}")
        return False

def testar_modelos_llm():
    """Testa os modelos LLM com uma pergunta simples."""
    print("\n🤖 Testando modelos LLM...")
    
    try:
        from src.models import ModelRunner, AVAILABLE_MODELS, GROQ_MODELS, GEMINI_MODELS
        from dotenv import load_dotenv
        
        # Carregar variáveis de ambiente
        load_dotenv()
        
        total_modelos_configurados = len(AVAILABLE_MODELS)

        print("   📋 Modelos disponíveis:")
        print(f"   🔓 Open Source (Groq): {len(GROQ_MODELS)} modelos")
        for modelo in GROQ_MODELS.keys():
            print(f"      - {modelo}")
        
        print(f"   🔒 Proprietários (Gemini): {len(GEMINI_MODELS)} modelos")
        for modelo in GEMINI_MODELS.keys():
            print(f"      - {modelo}")
        print(f"   📦 Total configurado no projeto: {total_modelos_configurados} modelos")
        
        # Pergunta simples para teste
        pergunta_teste = "What is Python programming language?"
        print(f"\n   ❓ Pergunta de teste: {pergunta_teste}")
        print("   " + "="*60)
        
        modelos_testados = 0
        modelos_funcionando = 0
        
        # Testar modelos Groq
        print("\n   🔓 Testando modelos Open Source (Groq):")
        for modelo_key in GROQ_MODELS.keys():
            try:
                print(f"\n   🧪 Testando {modelo_key}...")
                runner = ModelRunner(modelo_key)
                resposta = runner.generate(pergunta_teste)
                
                if resposta and not resposta.startswith('[ERRO]'):
                    print(f"   ✅ {modelo_key}: {resposta[:100]}...")
                    modelos_funcionando += 1
                else:
                    print(f"   ❌ {modelo_key}: {resposta}")
                
                modelos_testados += 1
                
            except Exception as e:
                print(f"   ❌ {modelo_key}: Erro - {str(e)[:100]}...")
                modelos_testados += 1
        
        # Testar modelos Gemini
        print("\n   🔒 Testando modelos Proprietários (Gemini):")
        for modelo_key in GEMINI_MODELS.keys():
            try:
                print(f"\n   🧪 Testando {modelo_key}...")
                runner = ModelRunner(modelo_key)
                resposta = runner.generate(pergunta_teste)
                
                if resposta and not resposta.startswith('[ERRO]'):
                    print(f"   ✅ {modelo_key}: {resposta[:100]}...")
                    modelos_funcionando += 1
                else:
                    print(f"   ❌ {modelo_key}: {resposta}")
                
                modelos_testados += 1
                
            except Exception as e:
                print(f"   ❌ {modelo_key}: Erro - {str(e)[:100]}...")
                modelos_testados += 1
        
        print(f"\n   📊 Resultado do teste:")
        print(f"   ✅ Modelos funcionando: {modelos_funcionando}/{modelos_testados}")
        print(f"   📈 Taxa de sucesso: {(modelos_funcionando/modelos_testados)*100:.1f}%")

        if modelos_testados != total_modelos_configurados:
            print(f"   ⚠️ Atenção: apenas {modelos_testados}/{total_modelos_configurados} modelos foram testados")
        else:
            print(f"   ✅ Cobertura de teste: {modelos_testados}/{total_modelos_configurados} modelos")
        
        if modelos_funcionando > 0:
            print("   🎉 Pelo menos um modelo está funcionando!")
            return True
        else:
            print("   ⚠️ Nenhum modelo está funcionando. Verifique as API keys.")
            return False
        
    except Exception as e:
        print(f"   ❌ Erro no teste de modelos: {e}")
        return False

def main():
    """Função principal do teste."""
    print("🚀 TESTE RÁPIDO DO SISTEMA")
    print("=" * 50)
    
    testes = [
        ("Imports", testar_imports),
        ("Estrutura de Arquivos", testar_estrutura_arquivos),
        ("Pasta de Resultados", testar_pasta_resultados),
        ("Sistema de Análise", testar_analisador),
        ("Modelos LLM", testar_modelos_llm)
    ]
    
    resultados = []
    
    for nome, teste in testes:
        try:
            resultado = teste()
            resultados.append((nome, resultado))
        except Exception as e:
            print(f"   ❌ Erro inesperado: {e}")
            resultados.append((nome, False))
    
    # Resumo
    print("\n" + "=" * 50)
    print("📋 RESUMO DOS TESTES")
    print("=" * 50)
    
    sucessos = 0
    for nome, resultado in resultados:
        status = "✅ PASSOU" if resultado else "❌ FALHOU"
        print(f"{nome}: {status}")
        if resultado:
            sucessos += 1
    
    print(f"\n🎯 Resultado: {sucessos}/{len(resultados)} testes passaram")
    
    if sucessos == len(resultados):
        print("🎉 Sistema funcionando perfeitamente!")
        print("\n💡 Próximos passos:")
        print("1. Execute: python main.py (múltiplas vezes)")
        print("2. Execute: python analysis/analysis.py")
    else:
        print("⚠️ Alguns testes falharam. Verifique os erros acima.")
    
    return sucessos == len(resultados)

if __name__ == "__main__":
    main()
