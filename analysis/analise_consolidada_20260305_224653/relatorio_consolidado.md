# 📊 Relatório Consolidado de Análise de Modelos LLM

**Data da Análise**: 05/03/2026 22:50:41
## 📊 Informações da Análise

| Métrica | Valor |
|:--------|------:|
| **Total de Respostas** | 567 |
| **Modelos Avaliados** | 7 |
| **Execuções Analisadas** | 3 |
| **Respostas Válidas** | 399 |
| **Taxa de Sucesso** | 70.4% |

**Metadados**: ✅ Timestamp, comprimento de prompt/resposta, flags de erro

## 📈 Resumo Executivo

❌ **Taxa de sucesso baixa**: 70.4% das respostas são válidas
🏆 **Melhor modelo acadêmico**: llama3_70b (score: 0.337)
📊 **Melhor modelo em consistência**: llama3_8b (taxa: 100.0%)
⚠️ **Modelos com problemas**: gemini_2_5_flash_lite (25.9%), gemini-2.0-flash-lite (0.0%)

## 🏆 Rankings Detalhados por Métrica

### BLEU

*Mede a similaridade entre texto gerado e referência (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 1.0000 | 1 |
| 🥈 | **llama3_8b** | 0.9956 | 2 |
| 🥉 | **gpt_oss_120b** | 0.5199 | 3 |
| 🏅 | **gpt_oss_20b** | 0.3955 | 4 |
| 🏅 | **qwen_32b** | 0.1462 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### ROUGE-1

*Mede sobreposição de palavras individuais (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 1.0000 | 1 |
| 🥈 | **llama3_8b** | 0.9724 | 2 |
| 🥉 | **gpt_oss_120b** | 0.8023 | 3 |
| 🏅 | **gpt_oss_20b** | 0.7257 | 4 |
| 🏅 | **qwen_32b** | 0.3761 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### ROUGE-2

*Mede sobreposição de bigramas (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 1.0000 | 1 |
| 🥈 | **llama3_8b** | 0.9720 | 2 |
| 🥉 | **gpt_oss_120b** | 0.7888 | 3 |
| 🏅 | **gpt_oss_20b** | 0.7846 | 4 |
| 🏅 | **qwen_32b** | 0.2812 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### ROUGE-L

*Mede sobreposição de subsequências mais longas (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 1.0000 | 1 |
| 🥈 | **llama3_8b** | 0.9810 | 2 |
| 🥉 | **gpt_oss_120b** | 0.9021 | 3 |
| 🏅 | **gpt_oss_20b** | 0.8344 | 4 |
| 🏅 | **qwen_32b** | 0.3881 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### BERTScore

*Mede similaridade semântica usando embeddings BERT (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_8b** | 1.0000 | 1 |
| 🥈 | **llama3_70b** | 0.9998 | 2 |
| 🥉 | **gpt_oss_120b** | 0.9812 | 3 |
| 🏅 | **gpt_oss_20b** | 0.9617 | 4 |
| 🏅 | **qwen_32b** | 0.9146 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### Respostas Válidas

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_8b** | 1.0000 | 1 |
| 🥈 | **llama3_70b** | 1.0000 | 2 |
| 🥉 | **qwen_32b** | 1.0000 | 3 |
| 🏅 | **gpt_oss_120b** | 0.9506 | 4 |
| 🏅 | **gpt_oss_20b** | 0.7160 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.2593 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### Taxa de Validade

*Percentual de respostas válidas (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_8b** | 1.0000 | 1 |
| 🥈 | **llama3_70b** | 1.0000 | 2 |
| 🥉 | **qwen_32b** | 1.0000 | 3 |
| 🏅 | **gpt_oss_120b** | 0.9506 | 4 |
| 🏅 | **gpt_oss_20b** | 0.7160 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.2593 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### Comprimento Médio

*Comprimento médio das respostas em caracteres*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **qwen_32b** | 1.0000 | 1 |
| 🥈 | **gemini_2_5_flash_lite** | 0.9272 | 2 |
| 🥉 | **llama3_70b** | 0.7965 | 3 |
| 🏅 | **llama3_8b** | 0.7856 | 4 |
| 🏅 | **gpt_oss_120b** | 0.3619 | 5 |
| 📊 | **gpt_oss_20b** | 0.2712 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### Palavras Médias

*Número médio de palavras por resposta*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **qwen_32b** | 1.0000 | 1 |
| 🥈 | **gemini_2_5_flash_lite** | 0.8362 | 2 |
| 🥉 | **llama3_8b** | 0.7311 | 3 |
| 🏅 | **llama3_70b** | 0.7265 | 4 |
| 🏅 | **gpt_oss_120b** | 0.3191 | 5 |
| 📊 | **gpt_oss_20b** | 0.2489 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### Consistência de Comprimento

*Consistência no tamanho das respostas (menor desvio é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **qwen_32b** | 1.0000 | 1 |
| 🥈 | **gemini_2_5_flash_lite** | 0.9947 | 2 |
| 🥉 | **llama3_70b** | 0.7213 | 3 |
| 🏅 | **llama3_8b** | 0.6900 | 4 |
| 🏅 | **gpt_oss_120b** | 0.3837 | 5 |
| 📊 | **gpt_oss_20b** | 0.1807 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

## 📊 Análise de Correlações entre Métricas

### Correlações Calculadas:
- **ROUGE-1 vs BERTScore**: 0.907
- **ROUGE-2 vs ROUGE-L**: 0.994
- **BLEU vs ROUGE-1**: 0.934

### Interpretação:
✅ **ROUGE-1 e BERTScore** têm alta correlação (consistência boa)
✅ **ROUGE-2 e ROUGE-L** têm alta correlação (consistência boa)


## 📊 Rankings Consolidados por Categoria

### Score Acadêmico

*Combinação de métricas de qualidade de texto (BLEU, ROUGE, BERTScore)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 1.0000 | 1 |
| 🥈 | **llama3_8b** | 0.9842 | 2 |
| 🥉 | **gpt_oss_120b** | 0.7988 | 3 |
| 🏅 | **gpt_oss_20b** | 0.7404 | 4 |
| 🏅 | **qwen_32b** | 0.4213 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### Score Evidently AI

*Métricas de qualidade e consistência das respostas*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **qwen_32b** | 1.0000 | 1 |
| 🥈 | **llama3_70b** | 0.8489 | 2 |
| 🥉 | **llama3_8b** | 0.8413 | 3 |
| 🏅 | **gemini_2_5_flash_lite** | 0.6553 | 4 |
| 🏅 | **gpt_oss_120b** | 0.5932 | 5 |
| 📊 | **gpt_oss_20b** | 0.4266 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### Score Geral

*Score final combinando todas as métricas com pesos balanceados*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 0.9244 | 1 |
| 🥈 | **llama3_8b** | 0.9128 | 2 |
| 🥉 | **qwen_32b** | 0.7106 | 3 |
| 🏅 | **gpt_oss_120b** | 0.6960 | 4 |
| 🏅 | **gpt_oss_20b** | 0.5835 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.3277 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

## 🔍 Análise Qualitativa

### 🎯 Modelo Mais Consistente: qwen_32b
- Menor variação no comprimento das respostas
- Maior estabilidade de performance

### 🧠 Modelo com Maior Fidelidade de Texto: llama3_8b
- Melhor similaridade semântica com referências
- Maior qualidade de conteúdo gerado

### 🛡️ Modelo Mais Confiável: llama3_8b
- Maior taxa de respostas válidas
- Menor incidência de erros

### 📝 Modelo Mais Detalhado: qwen_32b
- Respostas mais longas e detalhadas
- Maior riqueza de informação

### 📈 Análise de Correlações

- **Correlação Acadêmico vs Evidently AI**: 0.496
  - Correlação moderada: alguma relação entre métricas acadêmicas e qualidade de dados

### 🔓 vs 🔒 Open Source vs Proprietários

- **Score Médio Open Source**: 0.765
- **Score Médio Proprietários**: 0.164
- **Conclusão**: Modelos open source superam os proprietários em performance geral

## 🏆 Ranking dos Modelos

### 🥇 llama3_70b (Score: 0.3365)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0331
- **ROUGE-1**: 0.2928
- **ROUGE-2**: 0.1003
- **ROUGE-L**: 0.1648
- **BERTScore**: 0.6610

**Métricas Evidently AI:**
- **Respostas Válidas**: 81
- **Taxa de Validade**: 100.0%
- **Comprimento Médio**: 1015.7 ± 354.3 caracteres
- **Palavras Médias**: 147.6 ± 41.2

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 1.0000 (12/12)
- **HellaSwag Accuracy**: 1.0000 (9/9)

---

### 🥈 llama3_8b (Score: 0.3335)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0330
- **ROUGE-1**: 0.2847
- **ROUGE-2**: 0.0975
- **ROUGE-L**: 0.1616
- **BERTScore**: 0.6611

**Métricas Evidently AI:**
- **Respostas Válidas**: 81
- **Taxa de Validade**: 100.0%
- **Comprimento Médio**: 1001.9 ± 377.8 caracteres
- **Palavras Médias**: 148.5 ± 47.4

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.9167 (11/12)
- **HellaSwag Accuracy**: 1.0000 (9/9)

---

### 🥉 gpt_oss_120b (Score: 0.3083)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0172
- **ROUGE-1**: 0.2349
- **ROUGE-2**: 0.0791
- **ROUGE-L**: 0.1486
- **BERTScore**: 0.6487

**Métricas Evidently AI:**
- **Respostas Válidas**: 77
- **Taxa de Validade**: 95.1%
- **Comprimento Médio**: 461.5 ± 301.6 caracteres
- **Palavras Médias**: 64.8 ± 42.4

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.8333 (10/12)
- **HellaSwag Accuracy**: 1.0000 (9/9)

---

### 4º gpt_oss_20b (Score: 0.2856)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0131
- **ROUGE-1**: 0.2125
- **ROUGE-2**: 0.0787
- **ROUGE-L**: 0.1375
- **BERTScore**: 0.6358

**Métricas Evidently AI:**
- **Respostas Válidas**: 58
- **Taxa de Validade**: 71.6%
- **Comprimento Médio**: 345.9 ± 289.5 caracteres
- **Palavras Médias**: 50.6 ± 42.2

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.8333 (10/12)
- **HellaSwag Accuracy**: 1.0000 (9/9)

---

### 5º qwen_32b (Score: 0.2444)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0048
- **ROUGE-1**: 0.1101
- **ROUGE-2**: 0.0282
- **ROUGE-L**: 0.0639
- **BERTScore**: 0.6047

**Métricas Evidently AI:**
- **Respostas Válidas**: 81
- **Taxa de Validade**: 100.0%
- **Comprimento Médio**: 1275.3 ± 124.0 caracteres
- **Palavras Médias**: 203.2 ± 8.7

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.4167 (5/12)
- **HellaSwag Accuracy**: 0.8889 (8/9)

---

### 6º gemini_2_5_flash_lite (Score: 0.0130)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0000
- **ROUGE-1**: 0.0000
- **ROUGE-2**: 0.0000
- **ROUGE-L**: 0.0000
- **BERTScore**: 0.0000

**Métricas Evidently AI:**
- **Respostas Válidas**: 21
- **Taxa de Validade**: 25.9%
- **Comprimento Médio**: 1182.5 ± 120.6 caracteres
- **Palavras Médias**: 169.9 ± 11.9

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.0833 (1/12)
- **HellaSwag Accuracy**: 0.0000 (0/9)

---

### 7º gemini-2.0-flash-lite (Score: 0.0000)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0000
- **ROUGE-1**: 0.0000
- **ROUGE-2**: 0.0000
- **ROUGE-L**: 0.0000
- **BERTScore**: 0.0000

**Métricas Evidently AI:**
- **Respostas Válidas**: 0
- **Taxa de Validade**: 0.0%
- **Comprimento Médio**: 0.0 ± 0.0 caracteres
- **Palavras Médias**: 0.0 ± 0.0

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.0000 (0/12)
- **HellaSwag Accuracy**: 0.0000 (0/9)

---

## 📊 Análise Comparativa

**Ranking por Confiabilidade:**
🥇 **llama3_8b**: 100.0%

🥈 **llama3_70b**: 100.0%

🥉 **qwen_32b**: 100.0%

4º **gpt_oss_120b**: 95.1%

5º **gpt_oss_20b**: 71.6%

6º **gemini_2_5_flash_lite**: 25.9%

7º **gemini-2.0-flash-lite**: 0.0%

**Ranking por Comprimento de Resposta:**
🥇 **qwen_32b**: 1275.3 caracteres

🥈 **gemini_2_5_flash_lite**: 1182.5 caracteres

🥉 **llama3_70b**: 1015.7 caracteres

4º **llama3_8b**: 1001.9 caracteres

5º **gpt_oss_120b**: 461.5 caracteres

6º **gpt_oss_20b**: 345.9 caracteres

7º **gemini-2.0-flash-lite**: 0.0 caracteres

## 💡 Recomendações

### 🏆 Modelo Recomendado: llama3_70b

**Justificativa:**
- Melhor score composto considerando todas as métricas
- Equilíbrio entre precisão acadêmica e confiabilidade
- Boa performance em métricas de qualidade textual

### 🛡️ Modelo Mais Confiável: llama3_8b
- Taxa de respostas válidas: 100.0%

### 📝 Modelo Mais Detalhado: qwen_32b
- Comprimento médio: 1275.3 caracteres
