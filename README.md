# 🤖 Sistema de Comparação e Análise de Modelos LLM

Sistema completo para comparação de modelos de linguagem (LLM) com métricas acadêmicas, análise de qualidade de dados e relatórios consolidados.

## 📋 Resumo

Este projeto implementa um pipeline automatizado para testar e comparar diferentes modelos de linguagem através de prompts padronizados, calculando métricas de qualidade acadêmica (BLEU, ROUGE, BERTScore) e análises de qualidade de dados usando Evidently AI. O sistema gera relatórios detalhados por modelo e um relatório consolidado final.

## 🎯 Objetivo

- **Comparar performance** de diferentes modelos LLM em tarefas padronizadas
- **Avaliar qualidade** das respostas usando métricas acadêmicas reconhecidas
- **Analisar dados** com ferramentas profissionais de qualidade
- **Gerar relatórios** consolidados para tomada de decisão
- **Automatizar** todo o processo de teste e análise

## 📁 Estrutura do Projeto

```
LLM/
├── 📄 main.py                          # Ponto de entrada principal
├── 📄 requirements.txt                 # Dependências Python
├── 📄 README.md                        # Este arquivo
├── 📄 .env.example                     # Exemplo de variáveis de ambiente
│
├── 📁 src/                             # Código fonte principal
│   ├── 📄 config.py                    # Configurações centralizadas
│   ├── 📄 pipeline.py                  # Pipeline de execução dos LLMs
│   ├── 📄 utils.py                     # Utilitários e funções auxiliares
│   └── 📄 logger.py                    # Sistema de logging
│
├── 📁 analysis/                        # Sistema de análise
│   ├── 📄 analysis.py                  # Orquestrador principal de análise
│   ├── 📄 bleu_rouge.py               # Cálculo de métricas BLEU/ROUGE
│   ├── 📄 bertscore.py                # Cálculo de métricas BERTScore
│   └── 📄 evidently_reports.py        # Geração de relatórios Evidently AI
│
├── 📁 results/                         # Resultados das execuções
│   ├── 📁 resultado_1/                # Execução 1
│   ├── 📁 resultado_2/                # Execução 2
│   └── 📁 resultado_N/                # Execução N
│
└── 📁 analysis/                        # Análises consolidadas
    └── 📁 analise_consolidada_YYYYMMDD_HHMMSS/
        ├── 📄 relatorio_consolidado.md
        ├── 📁 modelo_X/
        │   ├── 📄 relatorio_modelo_X.md
        │   └── 📁 evidently_reports/
        │       ├── 📄 evidently_qualidade.html
        │       └── 📄 evidently_texto.html
        └── 📄 metricas_consolidadas.json
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
| `utils.py` | **Utilitários** - Funções para salvamento de dados, gerenciamento de pastas e formatação |
| `logger.py` | **Logging** - Sistema de logs estruturado para monitoramento e debug |

### Sistema de Análise (`analysis/`)

| Arquivo | Descrição |
|---------|-----------|
| `analysis.py` | **Orquestrador** - Classe principal que coordena toda a análise consolidada |
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
- **gemini_1_5_flash** - `models/gemini-1.5-flash`
- **gemini_2_5_flash_lite** - `models/gemini-2.5-flash-lite`

## 📊 Métricas Implementadas

### Métricas Acadêmicas
- **BLEU Score** - Precisão de n-gramas para avaliação de qualidade
- **ROUGE-1** - Sobreposição de unigramas
- **ROUGE-2** - Sobreposição de bigramas  
- **ROUGE-L** - Sobreposição de subsequência mais longa
- **BERTScore** - Similaridade semântica usando embeddings BERT

### Métricas de Qualidade (Evidently AI)
- **Distribuição de Comprimento** - Análise do tamanho das respostas
- **Contagem de Palavras** - Estatísticas de vocabulário
- **Qualidade do Texto** - Detecção de problemas de formatação
- **Drift de Dados** - Detecção de mudanças na distribuição
- **Métricas Estatísticas** - Média, mediana, desvio padrão, outliers

## 🔍 Sistema de Análise

### 1. Coleta de Dados
- Executa prompts padronizados contra todos os modelos
- Coleta respostas com metadados (timestamp, comprimento, flags de erro)
- Salva resultados em formato CSV e JSON

### 2. Processamento
- Consolida dados de múltiplas execuções por modelo
- Filtra respostas válidas vs. com erro
- Calcula métricas acadêmicas e de qualidade

### 3. Geração de Relatórios
- **Relatórios por Modelo** - Análise individual detalhada
- **Relatórios Evidently AI** - Análises de qualidade em HTML
- **Relatório Consolidado** - Comparação final com ranking

### 4. Ranking e Comparação
- Score composto baseado em métricas acadêmicas
- Fator de confiabilidade baseado em taxa de sucesso
- Penalizações para modelos com poucas respostas válidas

## 🚀 Como Usar

### Execução Completa
```bash
python main.py
```

### Execução da Análise Apenas
```bash
python -m analysis.analysis
```

### Configuração Personalizada
1. Edite `src/config.py` para ajustar modelos e parâmetros
2. Configure variáveis de ambiente no `.env`
3. Execute `python main.py`

## 📈 Exemplo de Saída

### Relatório Consolidado
```markdown
# 📊 Relatório Consolidado de Análise de Modelos LLM

**Data da Análise**: 04/01/2025 20:58:15
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
    peso_bleu = 0.2
    peso_rouge = 0.3
    peso_bertscore = 0.5
```

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

## 📝 Logs e Debug

O sistema gera logs detalhados em:
- Console durante execução
- Arquivos de log em `logs/` (se configurado)
- Relatórios de erro em `results/resultado_X/`

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