# 📊 Relatório Consolidado de Análise de Modelos LLM

**Data da Análise**: 06/03/2026 19:26:40
## 📊 Informações da Análise

| Métrica | Valor |
|:--------|------:|
| **Total de Respostas** | 567 |
| **Modelos Avaliados** | 7 |
| **Execuções Analisadas** | 3 |
| **Respostas Válidas** | 428 |
| **Taxa de Sucesso** | 75.5% |

**Metadados**: ✅ Timestamp, comprimento de prompt/resposta, flags de erro

## 📈 Resumo Executivo

❌ **Taxa de sucesso baixa**: 75.5% das respostas são válidas
🏆 **Melhor modelo acadêmico**: llama3_70b (score: 0.411)
📊 **Melhor modelo em consistência**: llama3_8b (taxa: 100.0%)
⚠️ **Modelos com problemas**: gemini_2_5_flash_lite (24.7%), gemini_3_flash_preview (24.7%)

## 🏆 Rankings Detalhados por Métrica

### BLEU

*Mede a similaridade entre texto gerado e referência (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 1.0000 | 1 |
| 🥈 | **llama3_8b** | 0.9723 | 2 |
| 🥉 | **gpt_oss_120b** | 0.4641 | 3 |
| 🏅 | **gpt_oss_20b** | 0.2594 | 4 |
| 🏅 | **qwen_32b** | 0.1836 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini_3_flash_preview** | 0.0000 | 7 |

### ROUGE-1

*Mede sobreposição de palavras individuais (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_8b** | 1.0000 | 1 |
| 🥈 | **llama3_70b** | 0.9934 | 2 |
| 🥉 | **gpt_oss_120b** | 0.7874 | 3 |
| 🏅 | **gpt_oss_20b** | 0.7026 | 4 |
| 🏅 | **qwen_32b** | 0.4375 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini_3_flash_preview** | 0.0000 | 7 |

### ROUGE-2

*Mede sobreposição de bigramas (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 1.0000 | 1 |
| 🥈 | **llama3_8b** | 0.9640 | 2 |
| 🥉 | **gpt_oss_120b** | 0.7130 | 3 |
| 🏅 | **gpt_oss_20b** | 0.6440 | 4 |
| 🏅 | **qwen_32b** | 0.3205 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini_3_flash_preview** | 0.0000 | 7 |

### ROUGE-L

*Mede sobreposição de subsequências mais longas (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 1.0000 | 1 |
| 🥈 | **llama3_8b** | 0.9990 | 2 |
| 🥉 | **gpt_oss_120b** | 0.8700 | 3 |
| 🏅 | **gpt_oss_20b** | 0.7972 | 4 |
| 🏅 | **qwen_32b** | 0.4316 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini_3_flash_preview** | 0.0000 | 7 |

### BERTScore

*Mede similaridade semântica usando embeddings BERT (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 1.0000 | 1 |
| 🥈 | **llama3_8b** | 0.9995 | 2 |
| 🥉 | **gpt_oss_120b** | 0.9712 | 3 |
| 🏅 | **gpt_oss_20b** | 0.9681 | 4 |
| 🏅 | **qwen_32b** | 0.9586 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini_3_flash_preview** | 0.0000 | 7 |

### Respostas Válidas

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_8b** | 1.0000 | 1 |
| 🥈 | **llama3_70b** | 1.0000 | 2 |
| 🥉 | **qwen_32b** | 1.0000 | 3 |
| 🏅 | **gpt_oss_120b** | 0.9180 | 4 |
| 🏅 | **gpt_oss_20b** | 0.8033 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini_3_flash_preview** | 0.0000 | 7 |

### Taxa de Validade

*Percentual de respostas válidas (0-1, maior é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_8b** | 1.0000 | 1 |
| 🥈 | **llama3_70b** | 1.0000 | 2 |
| 🥉 | **qwen_32b** | 1.0000 | 3 |
| 🏅 | **gpt_oss_120b** | 0.9180 | 4 |
| 🏅 | **gpt_oss_20b** | 0.8033 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini_3_flash_preview** | 0.0000 | 7 |

### Comprimento Médio

*Comprimento médio das respostas em caracteres*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **qwen_32b** | 1.0000 | 1 |
| 🥈 | **gemini_2_5_flash_lite** | 0.9398 | 2 |
| 🥉 | **llama3_70b** | 0.7873 | 3 |
| 🏅 | **llama3_8b** | 0.7784 | 4 |
| 🏅 | **gpt_oss_120b** | 0.3151 | 5 |
| 📊 | **gpt_oss_20b** | 0.2274 | 6 |
| 📊 | **gemini_3_flash_preview** | 0.0000 | 7 |

### Palavras Médias

*Número médio de palavras por resposta*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **qwen_32b** | 1.0000 | 1 |
| 🥈 | **gemini_2_5_flash_lite** | 0.8356 | 2 |
| 🥉 | **llama3_8b** | 0.7202 | 3 |
| 🏅 | **llama3_70b** | 0.7202 | 4 |
| 🏅 | **gpt_oss_120b** | 0.2752 | 5 |
| 📊 | **gpt_oss_20b** | 0.1981 | 6 |
| 📊 | **gemini_3_flash_preview** | 0.0000 | 7 |

### Consistência de Comprimento

*Consistência no tamanho das respostas (menor desvio é melhor)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **gemini_2_5_flash_lite** | 1.0000 | 1 |
| 🥈 | **qwen_32b** | 0.9878 | 2 |
| 🥉 | **gemini_3_flash_preview** | 0.7947 | 3 |
| 🏅 | **llama3_70b** | 0.6407 | 4 |
| 🏅 | **llama3_8b** | 0.6093 | 5 |
| 📊 | **gpt_oss_120b** | 0.1167 | 6 |
| 📊 | **gpt_oss_20b** | 0.0000 | 7 |

## 📊 Análise de Correlações entre Métricas

### Correlações Calculadas:
- **ROUGE-1 vs BERTScore**: 0.909
- **ROUGE-2 vs ROUGE-L**: 0.987
- **BLEU vs ROUGE-1**: 0.903

### Interpretação:
✅ **ROUGE-1 e BERTScore** têm alta correlação (consistência boa)
✅ **ROUGE-2 e ROUGE-L** têm alta correlação (consistência boa)


## 📊 Rankings Consolidados por Categoria

### Score Acadêmico

*Combinação de métricas de qualidade de texto (BLEU, ROUGE, BERTScore)*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 0.9987 | 1 |
| 🥈 | **llama3_8b** | 0.9870 | 2 |
| 🥉 | **gpt_oss_120b** | 0.7611 | 3 |
| 🏅 | **gpt_oss_20b** | 0.6743 | 4 |
| 🏅 | **qwen_32b** | 0.4663 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.0000 | 6 |
| 📊 | **gemini_3_flash_preview** | 0.0000 | 7 |

### Score Evidently AI

*Métricas de qualidade e consistência das respostas*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **qwen_32b** | 0.9976 | 1 |
| 🥈 | **llama3_70b** | 0.8296 | 2 |
| 🥉 | **llama3_8b** | 0.8216 | 3 |
| 🏅 | **gemini_2_5_flash_lite** | 0.5551 | 4 |
| 🏅 | **gpt_oss_120b** | 0.5086 | 5 |
| 📊 | **gpt_oss_20b** | 0.4064 | 6 |
| 📊 | **gemini_3_flash_preview** | 0.1589 | 7 |

### Score Geral

*Score final combinando todas as métricas com pesos balanceados*

| 🏆 | Modelo | Score | Rank |
|:---:|:-------|------:|:----:|
| 🥇 | **llama3_70b** | 0.9142 | 1 |
| 🥈 | **llama3_8b** | 0.9043 | 2 |
| 🥉 | **qwen_32b** | 0.7319 | 3 |
| 🏅 | **gpt_oss_120b** | 0.6349 | 4 |
| 🏅 | **gpt_oss_20b** | 0.5403 | 5 |
| 📊 | **gemini_2_5_flash_lite** | 0.2775 | 6 |
| 📊 | **gemini_3_flash_preview** | 0.0795 | 7 |

## 🔍 Análise Qualitativa

### 🎯 Modelo Mais Consistente: gemini_2_5_flash_lite
- Menor variação no comprimento das respostas
- Maior estabilidade de performance

### 🧠 Modelo com Maior Fidelidade de Texto: llama3_70b
- Melhor similaridade semântica com referências
- Maior qualidade de conteúdo gerado

### 🛡️ Modelo Mais Confiável: llama3_8b
- Maior taxa de respostas válidas
- Menor incidência de erros

### 📝 Modelo Mais Detalhado: qwen_32b
- Respostas mais longas e detalhadas
- Maior riqueza de informação

### 📈 Análise de Correlações

- **Correlação Acadêmico vs Evidently AI**: 0.534
  - Correlação moderada: alguma relação entre métricas acadêmicas e qualidade de dados

### 🔓 vs 🔒 Open Source vs Proprietários

- **Score Médio Open Source**: 0.745
- **Score Médio Proprietários**: 0.179
- **Conclusão**: Modelos open source superam os proprietários em performance geral

## 🏆 Ranking dos Modelos

### 🥇 llama3_70b (Score: 0.4108)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0405
- **ROUGE-1**: 0.3581
- **ROUGE-2**: 0.1247
- **ROUGE-L**: 0.2064
- **BERTScore**: 0.8451

**Métricas Evidently AI:**
- **Respostas Válidas**: 81
- **Taxa de Validade**: 100.0%
- **Comprimento Médio**: 1014.6 ± 353.1 caracteres
- **Palavras Médias**: 148.9 ± 40.3

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 1.0000 (12/12)
- **HellaSwag Accuracy**: 1.0000 (9/9)

---

### 🥈 llama3_8b (Score: 0.4102)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0394
- **ROUGE-1**: 0.3605
- **ROUGE-2**: 0.1202
- **ROUGE-L**: 0.2062
- **BERTScore**: 0.8447

**Métricas Evidently AI:**
- **Respostas Válidas**: 81
- **Taxa de Validade**: 100.0%
- **Comprimento Médio**: 1003.7 ± 371.8 caracteres
- **Palavras Médias**: 148.9 ± 44.9

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.9167 (11/12)
- **HellaSwag Accuracy**: 1.0000 (9/9)

---

### 🥉 gpt_oss_120b (Score: 0.3697)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0188
- **ROUGE-1**: 0.2839
- **ROUGE-2**: 0.0889
- **ROUGE-L**: 0.1796
- **BERTScore**: 0.8208

**Métricas Evidently AI:**
- **Respostas Válidas**: 76
- **Taxa de Validade**: 93.8%
- **Comprimento Médio**: 437.6 ± 316.1 caracteres
- **Palavras Médias**: 62.0 ± 44.3

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.8333 (10/12)
- **HellaSwag Accuracy**: 1.0000 (9/9)

---

### 4º gpt_oss_20b (Score: 0.3522)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0105
- **ROUGE-1**: 0.2533
- **ROUGE-2**: 0.0803
- **ROUGE-L**: 0.1646
- **BERTScore**: 0.8181

**Métricas Evidently AI:**
- **Respostas Válidas**: 69
- **Taxa de Validade**: 85.2%
- **Comprimento Médio**: 330.5 ± 266.3 caracteres
- **Palavras Médias**: 46.9 ± 36.8

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 1.0000 (12/12)
- **HellaSwag Accuracy**: 1.0000 (9/9)

---

### 5º qwen_32b (Score: 0.3141)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0074
- **ROUGE-1**: 0.1577
- **ROUGE-2**: 0.0400
- **ROUGE-L**: 0.0891
- **BERTScore**: 0.8101

**Métricas Evidently AI:**
- **Respostas Válidas**: 81
- **Taxa de Validade**: 100.0%
- **Comprimento Médio**: 1274.4 ± 127.6 caracteres
- **Palavras Médias**: 203.5 ± 9.8

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.5000 (6/12)
- **HellaSwag Accuracy**: 0.8889 (8/9)

---

### 6º gemini_2_5_flash_lite (Score: 0.0123)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0000
- **ROUGE-1**: 0.0000
- **ROUGE-2**: 0.0000
- **ROUGE-L**: 0.0000
- **BERTScore**: 0.0000

**Métricas Evidently AI:**
- **Respostas Válidas**: 20
- **Taxa de Validade**: 24.7%
- **Comprimento Médio**: 1200.8 ± 109.8 caracteres
- **Palavras Médias**: 171.4 ± 12.4

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.0000 (0/12)
- **HellaSwag Accuracy**: 0.0000 (0/9)

---

### 7º gemini_3_flash_preview (Score: 0.0123)

**Métricas Acadêmicas:**
- **BLEU Score**: 0.0000
- **ROUGE-1**: 0.0000
- **ROUGE-2**: 0.0000
- **ROUGE-L**: 0.0000
- **BERTScore**: 0.0000

**Métricas Evidently AI:**
- **Respostas Válidas**: 20
- **Taxa de Validade**: 24.7%
- **Comprimento Médio**: 52.6 ± 12.5 caracteres
- **Palavras Médias**: 8.2 ± 1.8

**Métricas de Benchmarks:**
- **MMLU Accuracy**: 0.0000 (0/12)
- **HellaSwag Accuracy**: 0.0000 (0/9)

---

## 📊 Análise Comparativa

**Ranking por Confiabilidade:**
🥇 **llama3_8b**: 100.0%

🥈 **llama3_70b**: 100.0%

🥉 **qwen_32b**: 100.0%

4º **gpt_oss_120b**: 93.8%

5º **gpt_oss_20b**: 85.2%

6º **gemini_2_5_flash_lite**: 24.7%

7º **gemini_3_flash_preview**: 24.7%

**Ranking por Comprimento de Resposta:**
🥇 **qwen_32b**: 1274.4 caracteres

🥈 **gemini_2_5_flash_lite**: 1200.8 caracteres

🥉 **llama3_70b**: 1014.6 caracteres

4º **llama3_8b**: 1003.7 caracteres

5º **gpt_oss_120b**: 437.6 caracteres

6º **gpt_oss_20b**: 330.5 caracteres

7º **gemini_3_flash_preview**: 52.6 caracteres

## 💡 Recomendações

### 🏆 Modelo Recomendado: llama3_70b

**Justificativa:**
- Melhor score composto considerando todas as métricas
- Equilíbrio entre precisão acadêmica e confiabilidade
- Boa performance em métricas de qualidade textual

### 🛡️ Modelo Mais Confiável: llama3_8b
- Taxa de respostas válidas: 100.0%

### 📝 Modelo Mais Detalhado: qwen_32b
- Comprimento médio: 1274.4 caracteres
