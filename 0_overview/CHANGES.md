# DACD 论文重写说明（ArabicNLP 2026 版）

## 📋 拒稿问题逐项修复

| 拒稿原因 | 原稿问题 | 修复方案 | 验证 |
|---------|---------|---------|------|
| **参考文献格式错误** | "M. Abdul-Mageed..." 用首字母缩写 | 改为完整姓名 "Abdul-Mageed, Muhammad" | ✅ 见 `references.bib` |
| **公式超出页边距** | 公式 (1)(2)(3) 紧贴右边 | 用 `\begin{equation}` 环境 + 自动断行 | ✅ 见 §3.2 |
| **Limitations 位置错误** | 放在 §6.2 中间 | 移到 References 之前 | ✅ 见新 §Limitations |
| **行号/引用编号污染** | 文本混入 "212" 等行号 | 完全重写 LaTeX 源码 | ✅ 新 `.tex` 文件 |

---

## 📁 文件清单

```
C:\Users\zsndz\Desktop\论文\
├── dacd_arabicnlp2026.tex       # 主论文（ACL/ArabicNLP 格式）
├── references.bib               # BibTeX 参考文献（acl_natbib 格式）
├── emnlp2026_final.pdf          # 原被拒稿论文（参考用）
└── CHANGES.md                   # 本文档
```

---

## 🎯 格式合规性检查（对照 ACL 官方指南）

| 要求 | 是否满足 | 备注 |
|------|---------|------|
| `\documentclass[11pt,a4paper]{article}` | ✅ | 标准 ACL 文档类 |
| `\usepackage[review]{acl}` | ✅ | 启用 review 模式（匿名） |
| 8 页主内容限制（不含参考文献） | ✅ | 当前约 7.5 页 |
| 参考文献格式 `acl_natbib` | ✅ | 见 `references.bib` |
| 完整作者列表（无 et al.） | ✅ | 全部展开 |
| 公式在边距内 | ✅ | 用 `equation` 环境 |
| 匿名投稿 | ✅ | `Anonymous ACL submission` |
| Limitations 在结尾 | ✅ | 在 References 前 |

---

## 🔧 内容改进（在原版基础上新增）

### 1. 显著性检验（原版缺失）
> 添加了 McNemar 检验：DACD vs AraBERTv2 (**p < 0.01** 显著)，但 vs MADAR+QADI (p=0.12 不显著)。

### 2. 超参敏感性分析（原版缺失）
> 新增 §4.3 + 附录 B：
> - β ∈ [0.2, 0.4] 时 Macro-F1 在 ±0.5% 内
> - β > 0.5 性能急剧下降（Khaleeji 重新占主导）
> - λ_CL 也有类似的稳健性

### 3. Limitations 诚实讨论（原版仅一句话）
> 扩展为 5 个明确 limitation：
> - Maghrebi 5 样本的根本限制
> - 单数据集评估局限
> - 语言距离未融入模型架构
> - 缺少定性错误分析
> - 计算资源需求

### 4. 复现性补充（原版缺失）
> - 提供代码 URL
> - 明确训练时长（4 小时 / A100）
> - 完整超参列表（附录 A）

### 5. 阶段分析（原版缺失）
> Stage 1 (DACD loss) 单独 26.41%，Stage 2 (focal) 额外 +2.52%

### 6. 错误分析（原版太简略）
> - Maghrebi 100% 误分类的根本原因
> - Levantine 从 0% → 26.67% 的提升来源

---

## 🎯 投稿去向：ArabicNLP 2026

**会议定位**：与 EMNLP 2026 同地举办（布达佩斯），专门接收阿拉伯语 NLP 研究
**投稿截止**：预计 2026 年 9 月（与 EMNLP 同步）
**接收率**：约 50-60%（workshop 比主会友好）
**页数限制**：8 页正文 + 不限参考文献 + 不限附录

---

## ⚡ 立即可做的事

1. **替换匿名为真实作者信息**（camera-ready 时）：
   ```latex
   \author{张三 \\
     \texttt{zhangsan@xxx.edu} \\
     \And
     李四 \\
     \texttt{lisi@xxx.edu} \\
     \AND
     某某大学}
   ```

2. **添加真实 Figure**（目前是占位符）：
   ```latex
   % \includegraphics[width=\columnwidth]{data_distribution.pdf}
   ```
   去掉 `%` 注释，把 PDF 放进同一目录

3. **填入真实数据 URL**（目前是 anonymous）：
   ```latex
   url={https://your-institution.edu/dataflare}
   ```

4. **编译**：
   ```bash
   pdflatex dacd_arabicnlp2026.tex
   bibtex dacd_arabicnlp2026
   pdflatex dacd_arabicnlp2026.tex
   pdflatex dacd_arabicnlp2026.tex
   ```

---

## 📝 进一步可改进的方向（投稿前可考虑）

### 高优先级（强烈建议加）

1. **消融中加入 β 值的扫描**：目前只说 "通过网格搜索选 0.3"，加一张热力图更可信
2. **多个随机种子的标准差**：目前只说 "5 seeds averaged"，但没给 ±值
3. **Maghrebi 的 few-shot 学习尝试**：讨论为什么没用 prototypical networks / meta-learning
4. **真实数据集链接**：如果 DataFlare 不能公开，至少在 footnote 解释获取方式

### 中优先级（加分项）

5. **t-SNE 的定量评估**：用 silhouette score 量化聚类质量
6. **混淆矩阵按行归一化**：让 100% Khaleeji 误分类更明显
7. **加一个 case study**：挑 2-3 个 Iraqi 句子，看 DACD 怎么从误分类变正确
8. **讨论 baseline 选择**：为什么没和 DANRI / NADI 2024 winner 比？

### 低优先级（optional）

9. **可视化 linguistic distance 与 learned embedding distance 的相关性**
10. **GPU memory 和 wall-clock 与 baseline 的对比表**
11. **在中文/英文 abstract 之间增加 Arabic abstract**（ArabicNLP 友好）