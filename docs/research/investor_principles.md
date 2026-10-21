# 经典投资思想：可量化、只定性、A 股适用

面向 zhiku 的研究备忘，不是投资建议。只写自己的概括；引文保持短句并标明出处。不确定的实证不写。每位后面有一句「对我们（中证800、月频、本地三年数据）是否做成公式」。

格林布拉特神奇公式和 Piotroski F-Score 仓库里已有实现，这里只补对照。

## 1. 本杰明·格雷厄姆

《证券分析》（Graham & Dodd, 1934）和《聪明的投资者》（Graham, 1949；修订 1973）把股票当成企业份额。核心主张：安全边际、Mr. Market 是报价服务不是导师、区分投资与投机。

可量化：净流动资产价值（NCAV = 流动资产 − 全部负债）低于市值约 2/3 的净净股；格雷厄姆数字 \(\sqrt{22.5 \times \mathrm{EPS} \times \mathrm{BPS}}\)（PE×PB 对 22.5 的经验上限）。Oppenheimer (1986) 在美股用格雷厄姆公开准则得到正超额。Lakonishok, Shleifer & Vishny (1994) 支持低估值反向策略。

只定性：管理层诚信、业务是否可理解。

A 股：Liu, Stambaugh & Yuan (2019) 表明中国价值主要在 EP 而不是 BM，且最小 30% 壳污染规模。中证800 里真正的净净股很少；格雷厄姆数字可用，净净股篮子可能为空。

**做成公式：** `graham`（NCAV/市值 + 格雷厄姆数字便宜程度）。预期：便宜、稀疏。过闸才进候选。

## 2. 沃伦·巴菲特 / 查理·芒格

巴菲特致股东信（Berkshire Hathaway 网站公开）：1986 年信定义 owner earnings（报告利润 − 维持性资本开支 + 非现金费用等）。反复强调能力圈、护城河、合理价格买好生意。芒格《穷查理宝典》与 1995 年演讲 “The Psychology of Human Misjudgment”：多元思维模型、避免标准误判。

可量化：高盈利能力、低杠杆、自由现金流优于应计利润。对应 Novy-Marx (2013) 毛利率溢价，以及 Asness, Frazzini & Pedersen (2019) Quality Minus Junk。Piotroski (2000) 的 F-Score 是把「基本面改善」写成 9 个是/否，已在本仓库。

只定性：护城河是否耐久、管理层资本配置、能力圈边界。芒格的心理清单不能从一张财报打分。

A 股：质量/盈利因子有跨国证据；本地三年样本上「高 ROE + 低负债」可算，护城河不能算。

**做成公式：** 不新开「巴菲特公式」（与已有 Piotroski / 质量门槛重叠）。进审核清单。

## 3. 菲利普·费雪

《怎样选择成长股》（Fisher, 1958）15 要点：销售增长潜力、研发、销售组织、利润率、劳工与管理关系、管理深度、成本分析、行业地位等。多数要调研。

可量化：长期高资本回报、利润率稳定；研发/收入若财报有「研发费用」则用，没有就记缺数据，不估。

只定性：15 点里关于管理层与销售组织的条目。

A 股：研发字段覆盖不全。高 ROE + 毛利率稳定接近质量因子（Novy-Marx 2013），不是费雪原意的全部。

**做成公式：** `fisher`（三年 ROE 高且毛利率稳定；有研发强度则加分）。缺研发不编造。

## 4. 彼得·林奇

《战胜华尔街》（Lynch, 1993）与《One Up on Wall Street》（1989）：六类公司（缓慢增长、稳定、快速增长、周期、转机、资产）和 PEG = PE / 盈利增长率。

可量化：PEG（需要盈利增速；用已公告净利润相对上一期，不用分析师预测）。六类里只有「便宜的增长」能写成截面规则。

只定性：用常识理解公司怎么赚钱、店铺/产品观察。

A 股：没有可靠的卖方一致预期面板。PEG 用已实现增速会落后一拍，且高速增长在 A 股常伴随高换手。没有把 PEG 当作顶级期刊单因子的把握，不当作已证实的 alpha。

**做成公式：** `lynch_peg`（PE>0、增速>0 的 PEG 越低越好）。过闸才进候选。

## 5. 霍华德·马克斯

《投资最重要的事》（Marks, 2011）与橡树备忘录（oaktreecapital.com）：周期位置、二阶思维、风险是损失的可能性不是波动。

可量化：粗周期——股票池近 12 个月收益偏高时减仓。这是启发式，不是马克斯的模型。

只定性：二阶思维、谈判力、信贷周期的叙事。

A 股：三年样本画不出完整信贷周期。周期位置只放进可选仓位模块，不写成选股公式。

**做成公式：** 不做选股。仓位模块里可选「热市减仓」。

## 6. 乔治·索罗斯

《金融炼金术》（Soros, 1987）反身性：价格改变基本面，基本面再改变价格。

可量化：没有稳定的单因子写法。强行用动量当反身性是附会。

只定性：识别反馈循环、何时证伪。

A 股：不适用做成公式。进审核时只问「估值与基本面是否互相加强」，缺叙事则缺数据。

**做成公式：** 不做。

## 7. 斯坦利·德鲁肯米勒

公开访谈与 Schwager《新市场奇才》（1992）章节：看流动性与宏观、看准了集中下注、错了就认。

可量化：错了就认 → 策略连续落后同池则降仓；集中 → 单票上限（与凯利一起）。不是选股公式。

只定性：流动性研判、宏观主题。

A 股：三年月频样本上「认错」可以回测；宏观流动性没有可靠免费长序列，不做假宏观因子。

**做成公式：** 不做选股。仓位模块：连续落后则降仓。

## 8. 利弗莫尔 / 利弗莫尔式交易

《股票作手回忆录》（Lefèvre, 1923）已进公有领域。要点：关键点、顺势、截断亏损。

可量化：趋势用 12−1 个月动量（Jegadeesh & Titman, 1993）；突破用相对 52 周高点的距离；止损放仓位层。

只定性：关键点是否「真突破」。

A 股：LSY (2019) 里短期反转显著、动量不是中国的主异象。趋势公式**预期难过闸**，仍要跑完再决定，不预设通过。

**做成公式：** `livermore`（12−1 动量 + 接近 52 周高点）。止损在仓位模块。

## 9. 威廉·欧奈尔 CAN SLIM

《How to Make Money in Stocks》（O'Neil，多版）：C/A 盈利加速，N 新意，S 供给，L 领导股，I 机构，M 市场。

可量化：盈利改善（ROA/利润同比）、不是新股（若有上市年限）、中等市值、相对强度（动量）、非 ST。I 与 M 没有机构持仓与大盘择时面板，记缺。

只定性：N（新产品/新管理）几乎全缺。

A 股：把 CAN SLIM 写成「盈利改善 + 动量」接近 Chan, Jegadeesh & Lakonishok (1996) 的动量与盈利。中国仍要面对反转强于动量。

**做成公式：** `canslim`（能算的字母打分，缺的字母不补）。

## 10. 乔尔·格林布拉特

《The Little Book That Beats the Market》（Greenblatt, 2005/2010）：高 ROC + 高 Earnings Yield。本仓库 `greenblatt` 已实现。特殊情况见《You Can Be a Stock Market Genius》（1997），属事项驱动，需要事件数据，本轮不做。

实证：方法是质量+价值的简单形式，与 Novy-Marx、Fama–French 盈利/价值因子同方向。样本外衰减的讨论很多，这里不引未核对的二手表。

A 股：已有实现，继续走闸门。

## 11. 瑞·达利欧

《原则》（Dalio, 2017）与桥水 All Weather：风险平价，按风险贡献配资产，不是选股。

可量化：Asness, Frazzini & Pedersen (2012) 讨论杠杆厌恶与风险平价。股票池内部做「全天候」缺少债券/商品腿。

只定性：原则中的决策文化。

A 股：本工具是股票选股，不做跨资产全天候。审核里只问杠杆与波动是否过高。

**做成公式：** 不做选股。

## 12. 爱德华·索普

Kelly (1956) 给出信息率与下注分数。Thorp (2006) 把凯利用到赌场与市场，并说明满凯利路径波动过大，实务常用不满凯利。

可量化：半凯利 \(f = \tfrac12 (p - (1-p)/b)\)，再加单票上限。p、b 用策略已实现月胜率与盈亏比，滚动估计，不用未来。

只定性：何时边有优势。

A 股：月频、成本 10bp+5bp 后，半凯利通常比满仓等权更保守。做成可选仓位模块并回测对比。

## 13. 吉姆·西蒙斯

Zuckerman (2019)《解开市场的市场》描述大奖章：短持有、统计套利、数据与执行，不是公开因子菜谱。书里没有可复现的官方年化表，这里不抄传闻数字。

可量化：对我们就是已有的量价模型；本轮不新增「西蒙斯公式」。

只定性：研究文化、执行。

A 股：三年日线复现不了大奖章。不做装饰性公式。

## 14. 纳西姆·塔勒布；事项驱动

「南秀姆」按常见译名对应 Nassim Nicholas Taleb。《黑天鹅》（2007）、《反脆弱》（2012）：尾部、杠铃（大量保守 + 少量凸性）、via negativa。

可量化：仓位上避用过高杠杆、单票上限；不做成「黑天鹅选股」。

事项驱动：格林布拉特 (1997) 的分拆/重组/破产，以及 Mitchell & Pulvino (2001) 风险套利。本地没有并购价差面板，**本轮不做事项驱动公式**。

只定性：脆弱性、隐性卖出期权。

## 15. 段永平 / 张磊

段永平：公开访谈反复讲商业模式、现金流、不懂不做（能力圈）。没有以他为名的期刊因子。可量化部分并入质量清单（ROE、经营现金流、负债）；「懂不懂」只能定性，缺调研记缺数据。

张磊《价值》（2020）：长期主义、生意本质、人才与组织。同样没有独立期刊因子。可量化部分与质量/巴菲特清单重叠；组织与人才缺数据。

A 股：两人的公开框架接近质量投资；不新开同名选股公式，避免装饰。进审核清单。

## 思想与文献对照（只列上面用到的）

| 思想 | 公开实证/方法 | 本工具 |
| --- | --- | --- |
| 格雷厄姆安全边际 / 净净股 | Oppenheimer 1986；LSV 1994；LSY 2019（中国价值在 EP） | `graham` |
| 质量 / 好生意 | Novy-Marx 2013；Asness et al. 2019 QMJ；Piotroski 2000 | 已有 F-Score + `fisher` + 审核 |
| 增长 / PEG | 林奇原书启发式；无把握写成已证实单因子 | `lynch_peg`，过闸才用 |
| 动量 / 趋势 / CAN SLIM | Jegadeesh & Titman 1993；Chan et al. 1996；中国以反转为主（LSY 2019） | `livermore` / `canslim` |
| 神奇公式 | Greenblatt 2005；质量+价值 | 已有 `greenblatt` |
| 周期 / 认错 / 止损 / 凯利 | Marks 2011；Thorp 2006；Kelly 1956 | 可选仓位，不进选股 |
| 反身性 / 大奖章 / 事项驱动 / 全天候 | 缺可复现面板或不是选股 | 不做公式 |

## 参考文献

1. Graham, B., & Dodd, D. (1934). *Security Analysis*. McGraw-Hill.
2. Graham, B. (1973). *The Intelligent Investor* (4th rev. ed.). Harper.
3. Oppenheimer, H. R. (1986). A test of Ben Graham’s stock selection criteria. *Financial Analysts Journal*, 42(5), 57–66.
4. Lakonishok, J., Shleifer, A., & Vishny, R. W. (1994). Contrarian investment, extrapolation, and risk. *Journal of Finance*, 49(5), 1541–1578.
5. Liu, J., Stambaugh, R. F., & Yuan, Y. (2019). Size and value in China. *Journal of Financial Economics*, 134(1), 48–69.
6. Berkshire Hathaway. *Letters to Shareholders*. https://www.berkshirehathaway.com/letters/letters.html
7. Novy-Marx, R. (2013). The other side of value: The gross profitability premium. *Journal of Financial Economics*, 108(1), 1–28.
8. Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019). Quality minus junk. *Review of Accounting Studies*, 24, 34–112.
9. Piotroski, J. D. (2000). Value investing: The use of historical financial statement information to separate winners from losers. *Journal of Accounting Research*, 38, 1–41.
10. Munger, C. T. (1995). The psychology of human misjudgment. Harvard Law School speech.
11. Fisher, P. A. (1958). *Common Stocks and Uncommon Profits*. Harper & Brothers.
12. Lynch, P. (1989). *One Up on Wall Street*. Simon & Schuster.
13. Lynch, P. (1993). *Beating the Street*. Simon & Schuster.
14. Marks, H. (2011). *The Most Important Thing*. Columbia University Press.
15. Soros, G. (1987). *The Alchemy of Finance*. Simon & Schuster.
16. Schwager, J. D. (1992). *The New Market Wizards*. HarperBusiness.
17. Lefèvre, E. (1923). *Reminiscences of a Stock Operator*. George H. Doran. Public domain.
18. Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65–91.
19. Chan, L. K. C., Jegadeesh, N., & Lakonishok, J. (1996). Momentum strategies. *Journal of Finance*, 51(5), 1681–1713.
20. O’Neil, W. J. *How to Make Money in Stocks*. McGraw-Hill (multiple eds.).
21. Greenblatt, J. (2005). *The Little Book That Beats the Market*. Wiley.
22. Greenblatt, J. (1997). *You Can Be a Stock Market Genius*. Simon & Schuster.
23. Dalio, R. (2017). *Principles*. Simon & Schuster.
24. Asness, C. S., Frazzini, A., & Pedersen, L. H. (2012). Leverage aversion and risk parity. *Financial Analysts Journal*, 68(1), 47–59.
25. Kelly, J. L., Jr. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4), 917–926.
26. Thorp, E. O. (2006). The Kelly criterion in blackjack, sports betting, and the stock market. In *Handbook of Asset and Liability Management*.
27. Zuckerman, G. (2019). *The Man Who Solved the Market*. Portfolio.
28. Taleb, N. N. (2007). *The Black Swan*. Random House.
29. Taleb, N. N. (2012). *Antifragile*. Random House.
30. Mitchell, M., & Pulvino, T. (2001). Characteristics of risk and return in risk arbitrage. *Journal of Finance*, 56(6), 2135–2175.
31. 张磊 (2020). 《价值》. 浙江教育出版社.
