# 🤖 Framework de Comparação de Modelos de Linguagem (LLMs)

<p align="center">
  <a href="https://github.com/caiocezarq/llm-comparison-benchmark">
    <img src="https://img.shields.io/badge/Status-Ativo-success?style=for-the-badge" alt="Status Ativo">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+">
  </a>
  <a href="https://console.groq.com/docs/overview">
    <img src="https://img.shields.io/badge/Groq-API-black?style=for-the-badge" alt="Groq API">
  </a>
  <a href="https://ai.google.dev/">
    <img src="https://img.shields.io/badge/Google%20Generative%20AI-Gemini-orange?style=for-the-badge&logo=google&logoColor=white" alt="Google Generative AI">
  </a>
  <a href="https://docs.evidentlyai.com/">
    <img src="https://img.shields.io/badge/EvidentlyAI-Enabled-purple?style=for-the-badge" alt="EvidentlyAI">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/Licença-MIT-yellow?style=for-the-badge" alt="Licença MIT">
  </a>
</p>

Framework para comparar e avaliar modelos de linguagem (LLMs) de forma objetiva, com métricas acadêmicas, benchmarks padronizados e relatórios estruturados.

---

## 🎯 Principais Recursos

- Execução automatizada de modelos com prompts padronizados
- Cálculo de métricas: BLEU, ROUGE e BERTScore
- Benchmarks: MMLU (conhecimento geral) e HellaSwag (senso comum)
- Análise de qualidade e consistência com EvidentlyAI
- Sistema de ranking comparativo com normalização min-max
- Relatórios por modelo e relatório consolidado final
- Estrutura modular e reprodutível

---

## 🧱 Arquitetura do Projeto

```text
llm-comparison-benchmark/
├── main.py                     # Execução principal
├── env_example.txt             # Exemplo de configuração
├── requirements.txt            # Dependências
│
├── src/
│   ├── config.py               # Configurações centrais
│   ├── pipeline.py             # Execução dos prompts
│   ├── models.py               # Wrappers para APIs
│   ├── utils.py                # Funções auxiliares
│   └── logger.py               # Sistema de logs
│
├── prompts/
│   ├── prompts.json            # Prompts estruturados
│   └── benchmarks.json         # Benchmarks MMLU / HellaSwag
│
├── analysis/
│   ├── analysis.py             # Consolidação das métricas
│   ├── ranking_system.py       # Geração de rankings
│   ├── bleu_rouge.py           # BLEU / ROUGE
│   ├── bertscore.py            # BERTScore
│   ├── mmlu.py                 # MMLU
│   ├── hellaswag.py            # HellaSwag
│   └── evidently_reports.py    # Relatórios EvidentlyAI
│
└── results/                    # Resultados versionados
```

---

## 🤖 Modelos Suportados

| Modelo | API | Categoria |
|---|---|---|
| LLaMA 3.x | Groq | Open Source |
| Qwen 3 | Groq | Open Source |
| GPT-OSS 20B / 120B | Groq | Open Weight |
| Gemini 2.5 Flash-Lite / Gemini 3 Flash Preview | Google Generative AI | Proprietário |

---

## ⚙️ Instalação

```bash
git clone https://github.com/caiocezarq/llm-comparison-benchmark
cd llm-comparison-benchmark
pip install -r requirements.txt
```

Configurar chaves:

```bash
cp env_example.txt .env
```

No Windows (PowerShell):

```powershell
Copy-Item env_example.txt .env
```

Preencha:

```env
GROQ_API_KEY=
GEMINI_API_KEY=
```

---

## 🚀 Como Executar

Executar teste rápido do ambiente e modelos:

```bash
python teste_rapido.py
```

Executar pipeline completo:

```bash
python main.py
```

Rodar somente análise:

```bash
python -m analysis.analysis
```

Gerar somente rankings:

```bash
python -m analysis.ranking_system
```

---

## 📊 Métricas e Benchmarks Implementados

| Tipo | Nome | Propósito |
|---|---|---|
| Similaridade Léxica | BLEU / ROUGE | Avalia proximidade linguística |
| Similaridade Semântica | BERTScore | Mede equivalência de significado |
| Conhecimento Geral | MMLU | Avalia entendimento multitarefa |
| Raciocínio de Senso Comum | HellaSwag | Avalia coerência contextual |
| Consistência de Texto | EvidentlyAI | Distribuição, drift e qualidade |

---

## 📁 Saídas do Sistema

```text
results/
  resultado_N/
    resultados_todos.csv
    resultados_todos.json
    resultados_[modelo].csv
    relatorio_pipeline.json
    relatorio_pipeline.txt
```

```text
analysis/
  analise_consolidada_YYYYMMDD_HHMMSS/
    relatorio_consolidado.md
    metricas_consolidadas.json
    normalized_metrics.json
    rankings.md
    modelo_[nome]/
      relatorio_[nome].md
      evidently_reports/*.html
```

---

## 🎓 Uso Acadêmico

✅ Ideal para artigos, dissertações e relatórios técnicos  
✅ Metodologia reprodutível  
✅ Resultados exportáveis e citáveis  
✅ Benchmarks amplamente utilizados na literatura

---

## 📌 Observações Metodológicas

- As métricas textuais (BLEU/ROUGE/BERTScore) são calculadas nos prompts abertos.
- Os benchmarks (MMLU/HellaSwag) são analisados separadamente por acurácia.
- Recomenda-se interpretar resultados em múltiplas execuções e considerar logs de erro/rate limit.

---

## 🤝 Contribuindo

```bash
git checkout -b feature/minha-feature
git commit -m "Descrição clara"
git push origin feature/minha-feature
```

Depois é só abrir um Pull Request.

---

## 📄 Licença

Distribuído sob licença MIT, com uso livre para fins acadêmicos e comerciais.
