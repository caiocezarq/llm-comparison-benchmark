# 📊 Relatório Consolidado de Análise de Modelos LLM

**Data da Análise**: 11/11/2025 21:38:27
## 📊 Informações da Análise

| Métrica | Valor |
|:--------|------:|
| **Total de Respostas** | 567 |
| **Modelos Avaliados** | 7 |
| **Execuções Analisadas** | 3 |
| **Respostas Válidas** | 489 |
| **Taxa de Sucesso** | 86.2% |

**Metadados**: ✅ Timestamp, comprimento de prompt/resposta, flags de erro

## 📈 Resumo Executivo

⚠️ **Boa taxa de sucesso**: 86.2% das respostas são válidas
🏆 **Melhor modelo acadêmico**: llama3_8b (score: 0.000)
📊 **Melhor modelo em consistência**: llama3_8b (taxa: 100.0%)
⚠️ **Modelos com problemas**: gemini-2.0-flash-lite (35.8%)

## 🏆 Rankings Detalhados por Métrica

### BLEU

*Mede a similaridade entre texto gerado e referência (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 1.0000 | 1 |
| 🥈 | **llama3_8b** | 0.9658 | 2 |
| 🥉 | **gpt_oss_20b** | 0.8191 | 3 |
| 🏅 | **gemini_2_5_flash_lite** | 0.7671 | 4 |
| 🏅 | **gpt_oss_120b** | 0.4942 | 5 |
| 📊 | **qwen_32b** | 0.1614 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### ROUGE-1

*Mede sobreposição de palavras individuais (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **gpt_oss_20b** | 1.0000 | 1 |
| 🥈 | **gpt_oss_120b** | 0.7961 | 2 |
| 🥉 | **llama3_8b** | 0.6986 | 3 |
| 🏅 | **gemini_2_5_flash_lite** | 0.6840 | 4 |
| 🏅 | **llama3_70b** | 0.6794 | 5 |
| 📊 | **qwen_32b** | 0.2637 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### ROUGE-2

*Mede sobreposição de bigramas (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 1.0000 | 1 |
| 🥈 | **llama3_8b** | 0.9892 | 2 |
| 🥉 | **gemini_2_5_flash_lite** | 0.9822 | 3 |
| 🏅 | **gpt_oss_120b** | 0.8057 | 4 |
| 🏅 | **gpt_oss_20b** | 0.6900 | 5 |
| 📊 | **qwen_32b** | 0.2675 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### ROUGE-L

*Mede sobreposição de subsequências mais longas (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **gpt_oss_20b** | 1.0000 | 1 |
| 🥈 | **gpt_oss_120b** | 0.7300 | 2 |
| 🥉 | **llama3_8b** | 0.4913 | 3 |
| 🏅 | **llama3_70b** | 0.4761 | 4 |
| 🏅 | **gemini_2_5_flash_lite** | 0.4626 | 5 |
| 📊 | **qwen_32b** | 0.1841 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### BERTScore

*Mede similaridade semântica usando embeddings BERT (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **gpt_oss_20b** | 1.0000 | 1 |
| 🥈 | **gpt_oss_120b** | 0.8968 | 2 |
| 🥉 | **llama3_8b** | 0.8956 | 3 |
| 🏅 | **gemini_2_5_flash_lite** | 0.8691 | 4 |
| 🏅 | **llama3_70b** | 0.8440 | 5 |
| 📊 | **qwen_32b** | 0.7882 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### Respostas Válidas

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_8b** | 1.0000 | 1 |
| 🥈 | **llama3_70b** | 1.0000 | 2 |
| 🥉 | **gpt_oss_120b** | 1.0000 | 3 |
| 🏅 | **gemini_2_5_flash_lite** | 1.0000 | 4 |
| 🏅 | **qwen_32b** | 1.0000 | 5 |
| 📊 | **gpt_oss_20b** | 0.5000 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### Taxa de Validade

*Percentual de respostas válidas (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_8b** | 1.0000 | 1 |
| 🥈 | **llama3_70b** | 1.0000 | 2 |
| 🥉 | **gpt_oss_120b** | 1.0000 | 3 |
| 🏅 | **gemini_2_5_flash_lite** | 1.0000 | 4 |
| 🏅 | **qwen_32b** | 1.0000 | 5 |
| 📊 | **gpt_oss_20b** | 0.5000 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### Comprimento Médio

*Comprimento médio das respostas em caracteres*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **qwen_32b** | 1.0000 | 1 |
| 🥈 | **gemini_2_5_flash_lite** | 0.8269 | 2 |
| 🥉 | **llama3_70b** | 0.7295 | 3 |
| 🏅 | **llama3_8b** | 0.7151 | 4 |
| 🏅 | **gemini-2.0-flash-lite** | 0.7083 | 5 |
| 📊 | **gpt_oss_120b** | 0.1131 | 6 |
| 📊 | **gpt_oss_20b** | 0.0000 | 7 |

### Palavras Médias

*Número médio de palavras por resposta*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **qwen_32b** | 1.0000 | 1 |
| 🥈 | **gemini_2_5_flash_lite** | 0.7531 | 2 |
| 🥉 | **llama3_70b** | 0.6604 | 3 |
| 🏅 | **llama3_8b** | 0.6529 | 4 |
| 🏅 | **gemini-2.0-flash-lite** | 0.6139 | 5 |
| 📊 | **gpt_oss_120b** | 0.0882 | 6 |
| 📊 | **gpt_oss_20b** | 0.0000 | 7 |

### Consistência de Comprimento

*Consistência no tamanho das respostas (menor desvio é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **qwen_32b** | 1.0000 | 1 |
| 🥈 | **gemini_2_5_flash_lite** | 0.8180 | 2 |
| 🥉 | **llama3_70b** | 0.6481 | 3 |
| 🏅 | **llama3_8b** | 0.6202 | 4 |
| 🏅 | **gemini-2.0-flash-lite** | 0.5611 | 5 |
| 📊 | **gpt_oss_120b** | 0.1960 | 6 |
| 📊 | **gpt_oss_20b** | 0.0000 | 7 |

## 📊 Análise de Correlações entre Métricas

### Correlações Calculadas:
- **ROUGE-1 vs BERTScore**: 0.863
- **ROUGE-2 vs ROUGE-L**: 0.617
- **BLEU vs ROUGE-1**: 0.816

### Interpretação:
✅ **ROUGE-1 e BERTScore** têm alta correlação (consistência boa)
⚠️ **ROUGE-2 e ROUGE-L** têm correlação moderada


## 📊 Rankings Consolidados por Categoria

### Score Acadêmico

*Combinação de métricas de qualidade de texto (BLEU, ROUGE, BERTScore)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **gpt_oss_20b** | 0.9018 | 1 |
| 🥈 | **llama3_8b** | 0.8081 | 2 |
| 🥉 | **llama3_70b** | 0.7999 | 3 |
| 🏅 | **gemini_2_5_flash_lite** | 0.7530 | 4 |
| 🏅 | **gpt_oss_120b** | 0.7446 | 5 |
| 📊 | **qwen_32b** | 0.3330 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.0000 | 7 |

### Score Evidently AI

*Métricas de qualidade e consistência das respostas*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **qwen_32b** | 1.0000 | 1 |
| 🥈 | **gemini_2_5_flash_lite** | 0.8796 | 2 |
| 🥉 | **llama3_70b** | 0.8076 | 3 |
| 🏅 | **llama3_8b** | 0.7976 | 4 |
| 🏅 | **gpt_oss_120b** | 0.4795 | 5 |
| 📊 | **gemini-2.0-flash-lite** | 0.3767 | 6 |
| 📊 | **gpt_oss_20b** | 0.2000 | 7 |

### Score Geral

*Score final combinando todas as métricas com pesos balanceados*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **gemini_2_5_flash_lite** | 0.8163 | 1 |
| 🥈 | **llama3_70b** | 0.8038 | 2 |
| 🥉 | **llama3_8b** | 0.8029 | 3 |
| 🏅 | **qwen_32b** | 0.6665 | 4 |
| 🏅 | **gpt_oss_120b** | 0.6120 | 5 |
| 📊 | **gpt_oss_20b** | 0.5509 | 6 |
| 📊 | **gemini-2.0-flash-lite** | 0.1883 | 7 |

## 🔍 Análise Qualitativa

### 🎯 Modelo Mais Consistente: qwen_32b
- Menor variação no comprimento das respostas
- Maior estabilidade de performance

### 🧠 Modelo com Maior Fidelidade de Texto: gpt_oss_20b
- Melhor similaridade semântica com referências
- Maior qualidade de conteúdo gerado

### 🛡️ Modelo Mais Confiável: llama3_8b
- Maior taxa de respostas válidas
- Menor incidência de erros

### 📝 Modelo Mais Detalhado: qwen_32b
- Respostas mais longas e detalhadas
- Maior riqueza de informação

### 📈 Análise de Correlações

- **Correlação Acadêmico vs Evidently AI**: 0.013
  - Correlação fraca: métricas acadêmicas e qualidade de dados são independentes

### 🔓 vs 🔒 Open Source vs Proprietários

- **Score Médio Open Source**: 0.687
- **Score Médio Proprietários**: 0.502
- **Conclusão**: Modelos open source superam os proprietários em performance geral

## 🏆 Ranking dos Modelos

### 🥇 gpt_oss_20b (Score: 0.3676)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0204
- **ROUGE-1**: 0.3233
- **ROUGE-2**: 0.0511
- **ROUGE-L**: 0.2674
- **BERTScore**: 0.8020

**Métricas Evidently AI:**
- **Respostas Válidas**: 55
- **Taxa de Validade**: 67.9%
- **Comprimento Médio**: 305.5 ± 250.9 caracteres
- **Palavras Médias**: 43.9 ± 35.5

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.0000 (0/12)
- **HellaSwag Accuracy**: 0.0000 (0/9)

---

### 🥈 gpt_oss_120b (Score: 0.3366)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0123
- **ROUGE-1**: 0.2574
- **ROUGE-2**: 0.0597
- **ROUGE-L**: 0.1952
- **BERTScore**: 0.7192

**Métricas Evidently AI:**
- **Respostas Válidas**: 81
- **Taxa de Validade**: 100.0%
- **Comprimento Médio**: 412.3 ± 280.2 caracteres
- **Palavras Médias**: 57.5 ± 38.7

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.0000 (0/12)
- **HellaSwag Accuracy**: 0.0000 (0/9)

---

### 🥉 llama3_8b (Score: 0.3228)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0241
- **ROUGE-1**: 0.2258
- **ROUGE-2**: 0.0733
- **ROUGE-L**: 0.1314
- **BERTScore**: 0.7182

**Métricas Evidently AI:**
- **Respostas Válidas**: 81
- **Taxa de Validade**: 100.0%
- **Comprimento Médio**: 981.0 ± 365.7 caracteres
- **Palavras Médias**: 144.2 ± 46.3

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.0000 (0/12)
- **HellaSwag Accuracy**: 0.0000 (0/9)

---

### 4º gemini_2_5_flash_lite (Score: 0.3146)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0191
- **ROUGE-1**: 0.2211
- **ROUGE-2**: 0.0728
- **ROUGE-L**: 0.1237
- **BERTScore**: 0.6970

**Métricas Evidently AI:**
- **Respostas Válidas**: 81
- **Taxa de Validade**: 100.0%
- **Comprimento Médio**: 1086.7 ± 249.6 caracteres
- **Palavras Médias**: 159.6 ± 29.8

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.0000 (0/12)
- **HellaSwag Accuracy**: 0.0000 (0/9)

---

### 5º llama3_70b (Score: 0.3105)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0250
- **ROUGE-1**: 0.2196
- **ROUGE-2**: 0.0741
- **ROUGE-L**: 0.1273
- **BERTScore**: 0.6769

**Métricas Evidently AI:**
- **Respostas Válidas**: 81
- **Taxa de Validade**: 100.0%
- **Comprimento Médio**: 994.7 ± 350.7 caracteres
- **Palavras Médias**: 145.4 ± 41.4

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.0000 (0/12)
- **HellaSwag Accuracy**: 0.0000 (0/9)

---

### 6º qwen_32b (Score: 0.2411)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0040
- **ROUGE-1**: 0.0852
- **ROUGE-2**: 0.0198
- **ROUGE-L**: 0.0492
- **BERTScore**: 0.6321

**Métricas Evidently AI:**
- **Respostas Válidas**: 81
- **Taxa de Validade**: 100.0%
- **Comprimento Médio**: 1250.2 ± 122.6 caracteres
- **Palavras Médias**: 197.5 ± 9.9

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.0000 (0/12)
- **HellaSwag Accuracy**: 0.0000 (0/9)

---

### 7º gemini-2.0-flash-lite (Score: 0.0179)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0000
- **ROUGE-1**: 0.0000
- **ROUGE-2**: 0.0000
- **ROUGE-L**: 0.0000
- **BERTScore**: 0.0000

**Métricas Evidently AI:**
- **Respostas Válidas**: 29
- **Taxa de Validade**: 35.8%
- **Comprimento Médio**: 974.7 ± 405.0 caracteres
- **Palavras Médias**: 138.2 ± 52.9

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.0000 (0/12)
- **HellaSwag Accuracy**: 0.0000 (0/9)

---

## 📊 Análise Comparativa

**Ranking por Confiabilidade:**
🥇 **llama3_8b**: 100.0%

🥈 **llama3_70b**: 100.0%

🥉 **gpt_oss_120b**: 100.0%

4º **qwen_32b**: 100.0%

5º **gemini_2_5_flash_lite**: 100.0%

6º **gpt_oss_20b**: 67.9%

7º **gemini-2.0-flash-lite**: 35.8%

**Ranking por Comprimento de Resposta:**
🥇 **qwen_32b**: 1250.2 caracteres

🥈 **gemini_2_5_flash_lite**: 1086.7 caracteres

🥉 **llama3_70b**: 994.7 caracteres

4º **llama3_8b**: 981.0 caracteres

5º **gemini-2.0-flash-lite**: 974.7 caracteres

6º **gpt_oss_120b**: 412.3 caracteres

7º **gpt_oss_20b**: 305.5 caracteres

## 💡 Recomendações

### 🏆 Modelo Recomendado: gpt_oss_20b

**Justificativa:**
- Melhor score composto considerando todas as métricas
- Equilíbrio entre precisão acadêmica e confiabilidade
- Boa performance em métricas de qualidade textual

### 🛡️ Modelo Mais Confiável: llama3_8b
- Taxa de respostas válidas: 100.0%

### 📝 Modelo Mais Detalhado: qwen_32b
- Comprimento médio: 1250.2 caracteres
