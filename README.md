# 🤖 Sistema de Comparação e Análise de Modelos LLM

Sistema completo para comparação de modelos de linguagem (LLM) com métricas acadêmicas, análise de qualidade de dados, sistema de ranking comparativo e relatórios consolidados.

## 📋 Resumo

Este projeto implementa um pipeline automatizado para testar e comparar diferentes modelos de linguagem através de prompts padronizados, calculando métricas de qualidade acadêmica (BLEU, ROUGE, BERTScore) e análises de qualidade de dados usando Evidently AI. O sistema gera relatórios detalhados por modelo, rankings comparativos e um relatório consolidado final com análise qualitativa.

## 🎯 Objetivo

- **Comparar performance** de diferentes modelos LLM em tarefas padronizadas
- **Avaliar qualidade** das respostas usando métricas acadêmicas reconhecidas
- **Testar benchmarks** padronizados (MMLU, HellaSwag) para avaliação comparativa
- **Analisar dados** com ferramentas profissionais de qualidade (Evidently AI)
- **Gerar rankings** comparativos com normalização e análise qualitativa
- **Automatizar** todo o processo de teste, análise e comparação
- **Fornecer insights** acadêmicos para tomada de decisão

## 📁 Estrutura do Projeto

```
LLMv3/
├── 📄 main.py                          # Ponto de entrada principal
├── 📄 requirements.txt                 # Dependências Python
├── 📄 README.md                        # Este arquivo
├── 📄 .env.example                     # Exemplo de variáveis de ambiente
│
├── 📁 src/                             # Código fonte principal
│   ├── 📄 config.py                    # Configurações centralizadas
│   ├── 📄 pipeline.py                  # Pipeline de execução dos LLMs
│   ├── 📄 models.py                    # Implementação dos modelos
│   ├── 📄 utils.py                     # Utilitários e funções auxiliares
│   └── 📄 logger.py                    # Sistema de logging
│
├── 📁 prompts/                         # Prompts e benchmarks
│   ├── 📄 prompts.json                 # 20 prompts padronizados estruturados
│   └── 📄 benchmarks.json              # Benchmarks padronizados
│
├── 📁 analysis/                        # Sistema de análise avançada
│   ├── 📄 __init__.py                  # Pacote de análise
│   ├── 📄 analysis.py                  # Orquestrador principal de análise
│   ├── 📄 ranking_system.py            # Sistema de ranking comparativo
│   ├── 📄 benchmarks.py                # Classe base para benchmarks
│   ├── 📄 mmlu.py                      # Calculadora MMLU
│   ├── 📄 hellaswag.py                 # Calculadora HellaSwag
│   ├── 📄 bleu_rouge.py               # Cálculo de métricas BLEU/ROUGE
│   ├── 📄 bertscore.py                # Cálculo de métricas BERTScore
│   └── 📄 evidently_reports.py        # Geração de relatórios Evidently AI
│
├── 📁 results/                         # Resultados das execuções
│   ├── 📁 resultado_1/                # Execução 1
│   ├── 📁 resultado_2/                # Execução 2
│   └── 📁 resultado_N/                # Execução N
│
├── 📁 analysis/                        # Análises consolidadas
│   └── 📁 analise_consolidada_YYYYMMDD_HHMMSS/
│       ├── 📄 relatorio_consolidado.md
│       ├── 📄 rankings.md              # Rankings comparativos
│       ├── 📄 normalized_metrics.json  # Métricas normalizadas
│       ├── 📄 generate_rankings.py     # Script de reprodução
│       ├── 📁 modelo_X/
│       │   ├── 📄 relatorio_modelo_X.md
│       │   ├── 📄 dados_modelo_X.csv
│       │   └── 📁 evidently_reports/
│       │       ├── 📄 evidently_qualidade.html
│       │       └── 📄 evidently_texto.html
│       └── 📄 metricas_consolidadas.json
│
└── 📁 docs/                            # Documentação
    └── 📄 DOCUMENTACAO_TCC_ANALISE_LLM.md
```

## ⚙️ Configurações Necessárias

### 1. Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# APIs dos Modelos
GROQ_API_KEY=sua_chave_groq_aqui
OPENAI_API_KEY=sua_chave_openai_aqui
GOOGLE_API_KEY=sua_chave_google_aqui

# Configurações de Execução
NUMERO_EXECUCOES=3
TIMEOUT_ENTRE_EXECUCOES=30
TIMEOUT_ENTRE_PERGUNTAS=3
```

### 2. Instalação de Dependências

```bash
pip install -r requirements.txt
```

### 3. Configuração de Modelos

Edite `src/config.py` para ajustar:
- Lista de modelos a testar
- Parâmetros de geração (max_tokens, temperature)
- Timeouts e limites de taxa
- Inclusão de benchmarks (INCLUDE_BENCHMARKS)

## 🏆 Benchmarks Padronizados

### **MMLU (Massive Multitask Language Understanding)**
- **Descrição**: Avaliação de conhecimento em múltiplas disciplinas
- **Métricas**: Accuracy por subject e geral
- **Implementação**: `analysis/mmlu.py`
- **Configuração**: Ativado via `INCLUDE_BENCHMARKS=True`

### **HellaSwag (Commonsense Reasoning)**
- **Descrição**: Raciocínio de senso comum e completamento de cenários
- **Métricas**: Accuracy geral
- **Implementação**: `analysis/hellaswag.py`
- **Configuração**: Ativado via `INCLUDE_BENCHMARKS=True`

### **Configuração de Benchmarks**
```python
# src/config.py
INCLUDE_BENCHMARKS = True  # Incluir benchmarks na execução
BENCHMARKS_FOLDER = "prompts"
BENCHMARKS_FILE = "benchmarks.json"
```

## 📄 Descrição dos Arquivos

### Arquivos Principais

| Arquivo | Descrição |
|---------|-----------|
| `main.py` | **Ponto de entrada** - Executa múltiplas execuções da pipeline e inicia análise consolidada |
| `requirements.txt` | **Dependências** - Lista de pacotes Python necessários |
| `.env.example` | **Template** - Exemplo de configuração de variáveis de ambiente |

### Código Fonte (`src/`)

| Arquivo | Descrição |
|---------|-----------|
| `config.py` | **Configurações** - Centraliza todas as configurações do sistema (modelos, timeouts, paths) |
| `pipeline.py` | **Pipeline Principal** - Executa prompts contra múltiplos LLMs e coleta respostas |
| `models.py` | **Modelos** - Implementação dos runners para diferentes APIs (Groq, OpenAI, Google) |
| `utils.py` | **Utilitários** - Funções para salvamento de dados, gerenciamento de pastas e formatação |
| `logger.py` | **Logging** - Sistema de logs estruturado para monitoramento e debug |

### Sistema de Análise (`analysis/`)

| Arquivo | Descrição |
|---------|-----------|
| `analysis.py` | **Orquestrador** - Classe principal que coordena toda a análise consolidada |
| `ranking_system.py` | **Sistema de Ranking** - Gera rankings comparativos com normalização e análise qualitativa |
| `bleu_rouge.py` | **Métricas BLEU/ROUGE** - Calcula métricas de qualidade de tradução/summarização |
| `bertscore.py` | **Métricas BERTScore** - Calcula similaridade semântica usando embeddings BERT |
| `evidently_reports.py` | **Relatórios Evidently** - Gera análises de qualidade de dados em HTML |

## 🤖 Modelos Utilizados

### Modelos Locais (Groq)
- **llama3_8b** - `llama-3.1-8b-instant`
- **llama3_70b** - `llama-3.3-70b-versatile`
- **qwen_32b** - `qwen/qwen3-32b`
- **deepseek_70b** - `deepseek-r1-distill-llama-70b`

### Modelos Open Source (Groq)
- **gpt_oss_20b** - `openai/gpt-oss-20b`
- **gpt_oss_120b** - `openai/gpt-oss-120b`

### Modelos Google (Gemini)
- **gemini_1_5_flash** - `models/gemini-1.5-flash` ⚠️ *Alta taxa de erro (49.5%)*
- **gemini_2_5_flash_lite** - `models/gemini-2.5-flash-lite`

## 📊 Métricas Implementadas

### Métricas Acadêmicas
- **BLEU Score** - Precisão de n-gramas para avaliação de qualidade
- **ROUGE-1** - Sobreposição de unigramas
- **ROUGE-2** - Sobreposição de bigramas (corrigido)
- **ROUGE-L** - Sobreposição de subsequência mais longa
- **BERTScore** - Similaridade semântica usando embeddings BERT

### Benchmarks Acadêmicos
- **MMLU** - Massive Multitask Language Understanding
- **HellaSwag** - Commonsense Reasoning

### Métricas de Qualidade (Evidently AI)
- **Distribuição de Comprimento** - Análise do tamanho das respostas
- **Contagem de Palavras** - Estatísticas de vocabulário
- **Qualidade do Texto** - Detecção de problemas de formatação
- **Drift de Dados** - Detecção de mudanças na distribuição
- **Métricas Estatísticas** - Média, mediana, desvio padrão, outliers

### Sistema de Ranking
- **Normalização Min-Max** - Escala 0-1 para comparação justa
- **Rankings Individuais** - Por cada métrica específica
- **Rankings Consolidados** - Por categoria (Acadêmico, Evidently AI, Geral)
- **Análise Qualitativa** - Insights e correlações entre métricas
- **Filtro Automático** - Exclusão de modelos com alta taxa de erro
- **Insights Executivos** - Geração automática de recomendações

## 🔍 Sistema de Análise

### 1. Coleta de Dados
- Executa 20 prompts padronizados e estruturados contra todos os modelos
- Coleta respostas com metadados (timestamp, comprimento, flags de erro)
- Salva resultados em formato CSV e JSON
- Detecta automaticamente modelos problemáticos

### 2. Processamento
- Consolida dados de múltiplas execuções por modelo
- Filtra respostas válidas vs. com erro
- Calcula métricas acadêmicas e de qualidade
- Executa benchmarks MMLU e HellaSwag
- Analisa correlações entre métricas

### 3. Sistema de Ranking
- **Normalização** de métricas para escala 0-1
- **Rankings individuais** por cada métrica
- **Rankings consolidados** por categoria
- **Análise qualitativa** com correlações e insights

### 4. Geração de Relatórios
- **Relatórios por Modelo** - Análise individual detalhada
- **Relatórios Evidently AI** - Análises de qualidade em HTML
- **Relatório Consolidado** - Comparação final com ranking
- **Rankings Comparativos** - Tabelas e análises normalizadas
- **Análise de Correlações** - Identificação de consistência entre métricas
- **Insights Executivos** - Recomendações automáticas

## 🚀 Como Usar

### Execução Completa
```bash
python main.py
```

### Execução com Benchmarks
```bash
# Ativar benchmarks em src/config.py
INCLUDE_BENCHMARKS = True
python main.py
```

### Execução da Análise Apenas
```bash
python -m analysis.analysis
```

### Execução do Sistema de Ranking
```bash
python -m analysis.ranking_system
```

### Configuração Personalizada
1. Edite `src/config.py` para ajustar modelos e parâmetros
2. Configure variáveis de ambiente no `.env`
3. Execute `python main.py`

## 📈 Exemplo de Saída

### Relatório Consolidado
```markdown
# 📊 Relatório Consolidado de Análise de Modelos LLM

**Data da Análise**: 06/01/2025 12:09:40
**Total de Respostas**: 480
**Modelos Testados**: 8
**Execuções**: resultado_1, resultado_2, resultado_3

## 🏆 Ranking dos Modelos

1. **qwen_32b** - Score: 8.45
2. **deepseek_70b** - Score: 8.32
3. **llama3_70b** - Score: 8.18
...

## 📊 Métricas por Modelo

### qwen_32b
- **BLEU Score**: 0.234
- **ROUGE-1**: 0.456
- **BERTScore**: 0.789
- **Taxa de Sucesso**: 95.8%
```

### Rankings Comparativos
```markdown
# 🏆 Rankings Comparativos de Modelos LLM

## Rankings por Métrica Individual

### BLEU
| Modelo | Score Normalizado | Rank |
|--------|------------------|------|
| qwen_32b | 0.8500 | 1 |
| deepseek_70b | 0.8200 | 2 |
...

## Rankings Consolidados por Categoria

### Score Acadêmico
| Modelo | Score | Rank |
|--------|-------|------|
| qwen_32b | 0.8450 | 1 |
| deepseek_70b | 0.8200 | 2 |
...
```

## 🛠️ Dependências

### Principais
- `pandas>=1.5.0` - Manipulação de dados
- `numpy>=1.21.0` - Computação numérica
- `transformers>=4.30.0` - Modelos BERT para BERTScore
- `evaluate>=0.4.0` - Métricas de avaliação
- `evidently[llm]>=0.7.14` - Análise de qualidade de dados

### APIs
- `groq>=0.9.0` - API Groq para modelos locais
- `openai>=1.0.0` - API OpenAI
- `google-generativeai>=0.8.0` - API Google Gemini

### Utilitários
- `beautifulsoup4>=4.12.0` - Parsing HTML
- `python-dotenv>=1.0.0` - Gerenciamento de variáveis de ambiente
- `requests>=2.28.0` - Requisições HTTP
- `scikit-learn>=1.3.0` - Normalização e análise estatística

## 🔧 Configurações Avançadas

### Ajuste de Modelos
```python
# src/config.py
MODELOS_GROQ = {
    "llama3_8b": "llama-3.1-8b-instant",
    "llama3_70b": "llama-3.3-70b-versatile",
    # Adicione novos modelos aqui
}
```

### Ajuste de Prompts
```python
# src/config.py
PROMPTS = [
    "O que é inteligência artificial e como ela está transformando o mundo?",
    "Explique a diferença entre machine learning e deep learning.",
    # Adicione novos prompts aqui
]
```

### Ajuste de Métricas
```python
# analysis/analysis.py
def _calcular_ranking_modelos(self, metricas_por_modelo):
    # Ajuste os pesos das métricas aqui
    peso_bleu = 0.15
    peso_rouge = 0.20
    peso_bertscore = 0.25
    peso_confiabilidade = 0.10
```

## 🆕 Funcionalidades Avançadas

### Sistema de Ranking Comparativo
- **Normalização automática** de métricas para comparação justa
- **Rankings individuais** por cada métrica específica
- **Rankings consolidados** por categoria (Acadêmico, Evidently AI, Geral)
- **Análise qualitativa** com correlações e insights
- **Filtro automático** de modelos problemáticos

### Análise Qualitativa
- **Modelo mais consistente** (menor variação)
- **Modelo com maior fidelidade** (melhor BERTScore)
- **Modelo mais confiável** (maior taxa de sucesso)
- **Análise de correlações** entre métricas acadêmicas e Evidently AI
- **Comparação Open Source vs Proprietários**
- **Insights executivos** automáticos

### Relatórios Avançados
- **Rankings em Markdown** com tabelas formatadas e emojis
- **Métricas normalizadas em JSON** para reprodutibilidade
- **Scripts de geração** para reproduzir análises
- **Análise de outliers** e distribuições estatísticas
- **Formatação visual** melhorada com descrições e contexto

### Benchmarks Acadêmicos
- **MMLU** - Avaliação de conhecimento geral
- **HellaSwag** - Avaliação de raciocínio de senso comum
- **Extração automática** de respostas A, B, C, D
- **Validação robusta** de respostas de múltipla escolha

## 🐛 Troubleshooting

### Problemas Comuns

1. **Erro de API Key**
   - Verifique se as chaves estão corretas no `.env`
   - Confirme se as APIs estão ativas

2. **Erro de Dependências**
   - Execute `pip install -r requirements.txt`
   - Verifique a versão do Python (3.8+)

3. **Erro de Memória**
   - Reduza `NUMERO_EXECUCOES` no config
   - Use modelos menores primeiro

4. **Rate Limits**
   - Aumente `TIMEOUT_ENTRE_PERGUNTAS`
   - Execute menos modelos por vez

5. **Erro no BERTScore**
   - Instale: `pip install bert-score`
   - Verifique se há GPU disponível para acelerar

6. **Erro no Evidently AI**
   - Instale: `pip install evidently[llm]`
   - Verifique se há dados suficientes para análise

7. **Problemas com ROUGE-2**
   - Sistema corrigido automaticamente
   - Verifique logs para debugging

8. **Modelos com alta taxa de erro**
   - Sistema detecta automaticamente
   - Modelos problemáticos são excluídos da análise principal

## 📝 Logs e Debug

O sistema gera logs detalhados em:
- Console durante execução
- Arquivos de log em `logs/` (se configurado)
- Relatórios de erro em `results/resultado_X/`
- Análises consolidadas em `analysis/analise_consolidada_*/`

## 🎓 Uso Acadêmico

### Para Trabalhos de TCC/Mestrado
- **Dados normalizados** em `normalized_metrics.json`
- **Rankings reproduzíveis** com scripts Python
- **Métricas acadêmicas** padronizadas (BLEU, ROUGE, BERTScore)
- **Benchmarks acadêmicos** (MMLU, HellaSwag)
- **Análise estatística** com Evidently AI
- **Análise de correlações** entre métricas
- **Relatórios detalhados** para documentação
- **Documentação TCC** completa em `docs/`

### Reproduzibilidade
- Scripts de geração automática de rankings
- Configurações centralizadas e versionadas
- Logs detalhados de execução
- Métricas normalizadas para comparação justa

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique a seção Troubleshooting
2. Consulte os logs de erro
3. Abra uma issue no repositório

---

**Desenvolvido com ❤️ para comparação e análise de modelos LLM**

*Sistema completo de avaliação acadêmica com métricas padronizadas, análise de qualidade de dados e ranking comparativo para pesquisa em modelos de linguagem.*