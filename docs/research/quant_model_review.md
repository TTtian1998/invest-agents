# 量化模型调研（中证800 / 月频 / 本地数据有限）

面向 zhiku 的选股打分，不是交易建议。只写核对过的论文或仓库；没核对到官方数字的模型不编 IC。

我们的约束：本地库大约 2023-10 起三年日线；中证800 有月度时点成分；行业字段只有证监会门类且覆盖不全；北向已停披露；组合是月度等权再平衡。下面每节最后一句写「对我们是否适用」。

## 1. A 股横截面异象与因子模型

Liu, Stambaugh & Yuan, 2019, “Size and Value in China”, *Journal of Financial Economics* 134(1): 48–69. [doi:10.1016/j.jfineco.2019.03.008](https://doi.org/10.1016/j.jfineco.2019.03.008)。作者主页提供 CH-3/CH-4 月度与日度因子：[finance.wharton.upenn.edu/~stambaug](https://finance.wharton.upenn.edu/~stambaug/)。

要点（论文正文，不是二手转述）：

- 规模：剔除最小 30%（壳价值）后再做 SMB，否则规模因子被壳污染。
- 价值：A 股用 EP（市盈率倒数）比 BM 更能吸收价值效应；简单复制 Fama–French (1993) 的 FF-3 解释不了 CH-3 的价值因子（FF-3 年化 alpha 约 16.8%）。
- CH-3（市场 + 调整后规模 + EP）能解释论文收集的 10 个 CAPM 异象中的 8 个；解释不了的主要是反转和换手。
- CH-4 再加 PMO（pessimistic-minus-optimistic，用异常换手度量情绪）后，反转和换手也被吸收。
- 文中异象类别包括：规模、价值、盈利、波动、收益反转、换手、投资、应计、非流动性。前六类对 CAPM 有显著多空；投资在中国对 CAPM 不显著。

**对我们：** 高度适用。中证800已经去掉了大量壳，但仍应避免用最小市值当「规模溢价」。EP、换手、短期反转、低波动是便宜且与月频匹配的特征；CH-4 的情绪因子我们缺历史异常换手面板，只能用换手率本身近似。预期：这几列比再堆 Alpha158 更可能抬截面 RankIC。代价：EP 要公告日对齐的盈利，未复权价才能算估值。

## 2. 机器学习资产定价

Gu, Kelly & Xiu, 2020, “Empirical Asset Pricing via Machine Learning”, *Review of Financial Studies* 33(5): 2223–2273. [doi:10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009)。

要点：在美股横截面比较多种 ML；树和神经网络最好，增益来自非线性交互；各方法都指向动量、流动性、波动这组主导信号。经济含义用多空组合 Sharpe 衡量，不是只看拟合 R²。

中国市场复现（能指到原文的）：

- Yang, 2020 年对 GKX 的代码复现，把样本换成中证500 时点成分、2010-01 至 2020-09 的月度收益，特征只用规模和账面市值比：[YangShuiShui/Article-Reproduce-Emperical-Asset-Pricing-via-Machine-Learning-RFS2020](https://github.com/YangShuiShui/Article-Reproduce-Emperical-Asset-Pricing-via-Machine-Learning-RFS2020)。这是方法移植，不是原论文的 A 股结果。
- Pei, 2020, “Empirical Asset Pricing via Machine Learning: Evidence from China”, ICEMGD 2020（Clausius Press）。用随机森林和 CSMAR 的 75 个公司特征，报告换手、行业动量、成交额最重要；取前 23 个特征后预测几乎不变、计算时间约少 68%。会议论文，样本与变量定义需自行核对，不当作顶级期刊结论。

**对我们：** 适用方向是「少而稳的主导信号 + 树模型」，不是上深度网络。GKX 用几十年美股和近百个特征；我们只有三年和几十列，神经网络/复杂集成预期收益低、过拟合代价高。优先 LightGBM 和极简单的线性截面组合。

## 3. qlib benchmarks（CSI300 / 共享股票池）

来源：[microsoft/qlib `examples/benchmarks/README.md`](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)（查阅时的主表）。标签是 qlib 默认日度，不是我们的月度换仓。数字是他们的实验，不是我们本地能复现的。

CSI300 + Alpha158（摘录报告较好的）：

| 模型 | IC | ICIR | Rank IC | Rank ICIR |
| --- | ---: | ---: | ---: | ---: |
| DoubleEnsemble (Zhang et al.) | 0.0521 | 0.4223 | 0.0502 | 0.4117 |
| XGBoost | 0.0498 | 0.3779 | 0.0505 | 0.4131 |
| LightGBM | 0.0448 | 0.3660 | 0.0469 | 0.3877 |
| TRA (Lin et al.) 全特征 | 0.0440 | 0.3535 | 0.0540 | 0.4451 |
| Linear | 0.0397 | 0.3000 | 0.0472 | 0.3531 |

我们本地 Alpha158 + LightGBM 只有 RankIC 0.026、RankICIR 0.177、Q5-Q1 +0.05%。对照上表，差距首先来自样本长度、标签和数据质量，不是「没上 TRA」。

qlib 仓库里还有 HIST、MASTER 的 example 目录：

- HIST：Xu, Liu, Chen, Xu, Xia, Li, 2021, “HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information”, [arXiv:2110.13716](https://arxiv.org/abs/2110.13716)。官方 Alpha158 主表没有 HIST 这一行；不在这里编 IC。
- MASTER：Li, Liu, Shen, Wang, Chen, Huang, 2024, “MASTER: Market-Guided Stock Transformer for Stock Price Forecasting”, AAAI 2024, [arXiv:2312.15235](https://arxiv.org/abs/2312.15235)，代码 [SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER)。论文在 CSI300/CSI800 上评自己的指标；qlib Alpha158 主表同样没有官方 RankICIR，这里不抄。

**对我们：** LightGBM / 线性是可落地的参照；DoubleEnsemble 是「同一个 LGB 的稳健集成」，预期中等、实现代价中等，排在特征与标签之后。TRA / HIST / MASTER 要序列模型和更长样本，本轮不做。qlib 表上的 RankICIR ~0.39 是日度、长样本的上界参照，不是我们三个月验证集该达到的数。

## 4. 公式化 alpha

- Kakushadze, 2016, “101 Formulaic Alphas”, *Wilmott* 2016(84): 72–80. [arXiv:1601.00991](https://arxiv.org/abs/1601.00991)。101 个可写进代码的实盘 alpha；持有期大约 0.6–6.4 **天**；alpha 之间平均相关约 15.9%。多数是量价短周期，不是月频基本面。
- 开源实现（从论文附录重写，不是 WorldQuant 官方库）：[lvlh2/alpha101](https://github.com/lvlh2/alpha101)。
- 国内券商：国泰君安证券 2014 年「191 个短周期价量 alpha」研报被多个开源因子库当作 `gtja191` 的来源（例如 [HKUDS/Vibe-Trading v0.1.8](https://github.com/HKUDS/Vibe-Trading/releases/tag/v0.1.8) 的说明）。原文是研报不是期刊，公式以开源复现为准，本仓库不整表搬运。

**对我们：** 101/191 的持有期是天，直接当月频标签会错配。有用的是其中已经变成低频的构件：反转、换手、波动、价量背离。本轮只试验 4 个低频构件（短反转、低换手、低波、EP），不上 101 全家桶。

## 5. 组合构建

- Grinold & Kahn, *Active Portfolio Management*, 2nd ed., McGraw-Hill, 2000。主动权重 ≈ 预测 alpha / 风险，强调换手和交易成本吃掉 IC。
- 风险模型：MSCI Barra 中国股票模型（CNE5 及后续）是商业风险模型，用行业 + 风格（规模、价值、动量、波动、流动性等）做中性化。我们没有 CNE5 暴露，只能做简版：证监会门类均值 + 对数规模回归残差。
- Ledoit & Wolf, 2004, “A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices”, *Journal of Multivariate Analysis* 88(2): 365–411。样本协方差在 N≈800、T 短时不可用，收缩估计才稳。本轮组合仍是等权 Top-30，不做协方差优化。

**对我们：** 月频等权 + 涨跌停/成本已经在回测引擎里。先把信号做成「对同池超额」，再谈中性化；完整优化器（换手约束、行业偏离、风险预算）预期收益有、实现代价高，放到信号 IC 起来之后。

## 本轮实验排序（预期收益 / 成本）

1. 标签改成 T+1 开盘 → T+20 收盘的同池超额（对齐月频）。
2. 加入 LSY 类低频信号：短反转、低换手、低波；有数据再加 EP。
3. 行业 + 规模中性化（简版）。
4. 月末抽样，减少重叠日度标签对 ICIR 的注水。
5. Walk-forward（只在验证段调是否采用）。
6. LambdaRank。
7. 全 A 训练、中证800 内评估（样本变宽，幸存者仍用时点过滤）。
8. 验证集上单独有效的改动再组合一次。

明确不做（本轮）：TRA / HIST / MASTER、完整 Alpha101、CNE5 优化器、5–8 年训练（本地日线从 2023-10 起，做不到）。

## 参考文献（只列正文用到的）

1. Liu, J., Stambaugh, R. F., & Yuan, Y. (2019). Size and value in China. *Journal of Financial Economics*, 134(1), 48–69. https://doi.org/10.1016/j.jfineco.2019.03.008
2. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009
3. Yang, Q. (2020). Article reproduce: Empirical asset pricing via machine learning (RFS 2020). https://github.com/YangShuiShui/Article-Reproduce-Emperical-Asset-Pricing-via-Machine-Learning-RFS2020
4. Pei, Y. (2020). Empirical asset pricing via machine learning: Evidence from China. *ICEMGD 2020*. https://www.clausiuspress.com/conferences/LNEMSS/ICEMGD%202020/365.pdf
5. Microsoft Qlib. Benchmarks README. https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md
6. Xu, W., Liu, W., Chen, C., Xu, J., Xia, Q., & Li, J. (2021). HIST. arXiv:2110.13716.
7. Li, T., Liu, Z., Shen, Y., Wang, X., Chen, H., & Huang, S. (2024). MASTER. *AAAI*. arXiv:2312.15235. https://github.com/SJTU-DMTai/MASTER
8. Kakushadze, Z. (2016). 101 formulaic alphas. *Wilmott*, 2016(84), 72–80. arXiv:1601.00991
9. Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management* (2nd ed.). McGraw-Hill.
10. Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
