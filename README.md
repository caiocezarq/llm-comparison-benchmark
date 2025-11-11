# 🤖 Framework de Comparação de Modelos de Linguagem (LLMs)

<p align="center">

<a>
  <img src="https://img.shields.io/badge/Status-Ativo-success?style=for-the-badge">
</a>
<a>
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white">
</a>
<a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/Licença-MIT-yellow?style=for-the-badge">
</a>
<a>
  <img src="https://img.shields.io/badge/EvidentlyAI-Enabled-purple?style=for-the-badge">
</a>

</p>

Framework desenvolvido para **comparar e avaliar modelos de linguagem (LLMs)** de forma **objetiva**, utilizando **métricas acadêmicas**, **benchmarks padronizados** e **relatórios estruturados**.  
Ideal para **pesquisa acadêmica**, **análise de performance** e **seleção de modelos para produção**.

---

## 🎯 Principais Recursos

- Execução automatizada de modelos usando prompts padronizados
- Cálculo de métricas: **BLEU**, **ROUGE**, **BERTScore**
- Benchmarks: **MMLU** (conhecimento geral) e **HellaSwag** (senso comum)
- Análise de distribuição e consistência com **EvidentlyAI**
- Sistema de **ranking comparativo** (normalização min-max)
- Relatórios **por modelo** + **Consolidado final**
- Estrutura modular com **reprodutibilidade garantida**

---

## 🧱 Arquitetura do Projeto

```
llm-comparison-benchmark/
├── main.py                     # Execução principal
├── .env.example                # Exemplo de configuração
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
| **LLaMA 3.x** | Groq | Open Source |
| **Qwen 3** | Groq | Open Source |
| **GPT-OSS 20B / 120B** | OpenAI | Open Weight |
| **Gemini Flash / Flash-Lite** | Google AI | Proprietário |

---

## ⚙️ Instalação

```bash
git clone https://github.com/caiocezarq/llm-comparison-benchmark
cd llm-comparison-benchmark
pip install -r requirements.txt
```

Configurar chaves:

```bash
cp .env.example .env
```

Preencha:

```
GROQ_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
```

---

## 🚀 Como Executar

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
| Similaridade Léxica | **BLEU / ROUGE** | Avalia proximidade linguística |
| Similaridade Semântica | **BERTScore** | Mede equivalência de significado |
| Conhecimento Geral | **MMLU** | Avalia entendimento multitarefa |
| Raciocínio de Senso Comum | **HellaSwag** | Avalia coerência contextual |
| Consistência de Texto | **EvidentlyAI** | Distribuição, drift e qualidade |

---

## 📁 Saídas do Sistema

```
results/
└── resultado_YYYYMMDD_HHMMSS/
    ├── respostas.csv
    ├── metricas_normalizadas.json
    ├── relatorio_consolidado.md
    └── modelo_X/
         ├── relatorio_individual.md
         └── evidently_reports/*.html
```

---

## 🎓 Uso Acadêmico

✔ Ideal para **TCC, artigos, dissertações e relatórios técnicos**  
✔ Metodologia reprodutível  
✔ Resultados exportáveis e citáveis  
✔ Benchmarks amplamente utilizados na literatura

---

## 🤝 Contribuindo

```bash
git checkout -b feature/minha-feature
git commit -m "Descrição clara"
git push origin feature/minha-feature
```

Depois é só abrir um **Pull Request**.

---

## 📄 Licença

Distribuído sob licença **MIT** — uso livre para fins acadêmicos e comerciais.
