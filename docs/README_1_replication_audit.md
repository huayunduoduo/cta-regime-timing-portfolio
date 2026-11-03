# 华泰时序CTA：研报规则的日频复现与差异审计

复现2026-06-10华泰证券《时序CTA方法论综述：市场状态、开仓信号与退出机制》的三资产单资产实验。

**状态：公开规则已经实现并实际运行，但未成功复现原报告的业绩。** 本地回测为2020—2025年，原报告到2026年5月；采用日频固定止损撮合，非分钟逐笔复原。未披露的规则全部列于 [RULES.md](RULES.md)。不包含宏观、配置或ES增量。

## 已包含

- 沪深300ETF 510300、黄金ETF 518880、国债ETF 511010，多空双向理想化账户。
- ER30自适应21—250日AR-GARCH、MAD/Huber清洗、1000次有效bootstrap、10日状态确认。
- 完整策略、恒趋势、恒震荡、无ATR（反向信号退出假设）和买入持有，共15个主结果。
- 无ATR字面删除、Wilder RSI、14日局部TR截断、无单向约束吊灯、零成本、成本翻倍，共18个敏感性结果。
- 次日开盘执行、ATR吊灯止损、跳空撮合、RSI40/60退出、每次1‰滑点+1‰手续费。
- 现金/库存/分红应收及支付账本、信号、成交、平仓交易、逐年指标、自包含HTML与核验记录。

## 运行

Python 3.10+。Windows可用conda；从头估计和模型测试需要可用的C++17编译器 `g++`（例如MSYS2 MinGW-w64，加入PATH）。Linux同样需要g++。

```bash
python -m pip install -r requirements.txt
python run.py --data /path/to/data
python verify_cache.py --data /path/to/data
python validate_results.py --data /path/to/data
python -m unittest discover -s tests -v
python build_report.py
```

`data`目录需放自己的 `etf_daily.csv` 和 `etf_dividend.csv`，格式见 [DATA.md](DATA.md)。路径可用Windows引号，例如 `--data "D:/cta/data"`。原始行情和研报PDF均不随GitHub包分发，不提供无授权再分发承诺。

提供已计算模型缓存。输入DataFrame哈希一致时复用8,848个窗口，不降低bootstrap次数；3个抽检窗口本次均独立重算、1000次有效结果与缓存完全一致。**这不是本轮重新完成8,848个估计。** 原始数据、行顺序或数值改变会使缓存失效，重新生成。

全量重新估计（耗时显著更长）：

```bash
python run.py --data /path/to/data --rebuild-models --workers 4
```

可用 `--start` 和 `--end` 指定测试日期，但不得把反复查看过的历史重新称作未见样本。无超参数选择，不按年挑最优。本项目是固定参数历史复现，不是训练/验证择优后真正独立的样本外检验。

## 结果入口

- `results/report.html`：打开即用，图片内嵌，无外部依赖。
- `results/summary.csv`：15个主结果；`yearly.csv`：逐年结果（包括字面无ATR对照）。
- `results/sensitivity.csv`：18个敏感性结果。
- `{code}_signals.csv`：所有预热与测试日的模型、CI、状态、指标。
- `{code}_{variant}_nav.csv / fills.csv / trades.csv`：日账本、所有成交、已平仓交易。
- `model_audit.csv / cache_recompute_audit.json / ledger_audit.json / manifest.json`：数据哈希、缓存核对、账本核对、环境与假设。
- `paper_reference.csv`：原文结果，区间不同，只供溯源，不作等区间绩效差值比较。

本实现的空头不含券源、融券利息、保证金与利息，不代表三只ETF可无条件实盘做空。现金与闲置分红利率设0。末日按市值记账，不人为平仓提高或降低交易胜率。胜率和盈亏比均来自费用后已平仓交易。
完整计划见ROADMAP.md。当前复现未重现研报绩效，以下均为待实施、待验证的增量，不代表已取得收益改善。

研究问题：宏观信息、风险配置与VaR/ES约束，能否在扣除交易成本后改善股债金组合的风险收益与持有体验。主组合为沪深300ETF（510300）、黄金ETF（518880）、国债ETF（511010）；三个股票指数只作替代检验，避免重复权益暴露放大组合风险。

模块	研究内容
多资产配置	等权、逆波动率、约束风险平价，计入权重漂移、调仓与成本
风险预测	历史模拟、EWMA正态、过滤历史模拟的一日VaR/ES；检验覆盖率、违约聚集与尾部校准
ES约束	用资产联合历史冲击估组合ES，超预算收缩整体暴露，并固定重新加仓规则
宏观状态	先分析增长、流动性环境下CTA表现，再检验有限幅度的风险预算调整
对照版本：A股债金静态等权、B三个CTA策略等权、C固定CTA＋风险配置、D＝C＋宏观风险预算、E＝C＋ES约束、F＝C＋宏观＋ES。另设简单降仓对照，识别改善是否仅来自风险暴露下降。

实施顺序：先修正分红复权与宏观时点、建立日频基线，再做配置对照与风险预测校准，然后分别验证ES约束与宏观预算，最后联合实验。评价按时间滚动训练/验证/测试，禁止用测试结果反向选择参数；保留逐年、分阶段与失败结果。
## 复现差异诊断（本轮新增）

```bash
python audit.py --data /path/to/data --optimizer-check
python build_audit_report.py
```

入口：`results/audit/audit_report.html`。包含状态条件通过率、实际净值盈亏归因、黄金主要亏损交易、相同规则仅多头/仅空头对照，以及15个窗口的替代求解器抽查。完整多空基线逐日重放与原交付一致，未更改阈值或交易规则。方向对照只用于归因，不声称为研报策略或已验证的增量。

主要发现：黄金趋势日仅3.78%，震荡空头是主要亏损来源；但仅多头年化0.35%，仍未跟上买入持有19.18%。国债零成本年化约−0.10%，研报费用后−5.49%。一个沪深300抽样窗口存在更优似然解，数值稳健性仍待扩大核查，未据此擅自改写全部模型缓存。当前结论仍为未成功复现原文业绩。

# 新增：状态识别改造与逐项对照

本包在原工程上新增 `improvement_signals.py`、`run_improvements.py` 和 `tests/test_improvements.py`。原 `engine.py`、AR-GARCH、bootstrap、原回测程序及历史结果不修改。下方原README仍可用于研报复现；**新的研究入口是run_improvements.py**，输出在独立的 `results/improvements/`。

## 为什么改

原规则把“没有通过趋势确认”直接当作震荡，再用RSI反向开仓。这不能区分明确震荡与证据不足；同时ρ>0刻画短期收益惯性，不保证能识别长期上涨。连续10日确认可能过于迟钝；ρ的t值和sigma用OR连接也不等于严格排除高波动。

本次不追求对齐研报收益表，而研究这些条件是否有增量。所有方案固定参数、一视同仁用于股债金，完整保留失败结果，不按资产挑最好的版本。

## 已实现的8组实验

| 版本名 | 行为 | 对照目的 |
|---|---|---|
| baseline_full | 原多空完整策略，趋势+RSI反转 | 原始参考，不与仅多头方案直接作单因素归因 |
| baseline_long | 原规则，但负向入场全部屏蔽 | 同方向权限的公平基线 |
| neutral_10 | 保留原趋势判定；非趋势不再执行RSI反转 | 与baseline_long比较是否需要非趋势反转 |
| no_rho_10 | 在neutral_10上取消ρ>0硬门槛 | 单独检验ρ门槛 |
| no_rho_3 | 在no_rho_10上将确认由连续10日改为连续3日 | 检验缩短确认 |
| no_rho_3of5 | 与no_rho_10平行；最近完整5日中至少3日同方向且当日证据同方向 | 检验容忍短暂中断，不能与no_rho_3称作唯一变量对照 |
| risk_separate_3of5 | 在no_rho_3of5上，将风险条件改为sigma历史分位数<95% | 单独检验高波动过滤；本轮尚未实现连续仓位缩放 |
| buyhold | 原账本买入持有 | 简单基准 |

除baseline_full外，全部为多头/现金，不融券、不加杠杆。不确定或下降趋势不新开多仓；**状态失效不会自动卖出已有仓位，已有趋势单仍由原ATR吊灯退出**。这是为了保持退出规则一致，不等于状态不确定时立即全仓现金。日内不重入，盘前固定止损；成交、分红、费用完全复用原engine。

`neutral_10`和`no_rho_10/3`的连续确认按原代码定义：条件连续满足即可，方向取当日TSI；原代码并未要求整个确认段TSI符号一致。`3of5`明确要求至少3天同方向，且当天有同方向证据。所有判定仅使用截至当日收盘的数据，次日开盘执行。

保留原TSI条件：CI不跨0或abs(TSI)历史分位数>50%；后者只表示历史相对强度，不是显著性检验。`no_rho_*`仍保留原第三条OR风险条件，直到`risk_separate_3of5`才独立改它。确认10日/3日/3of5是事先给定的小集合，没有遍历搜索收益最高的天数。

## 运行（解压后在工程目录执行）

```bash
python -m pip install -r requirements.txt
python run_improvements.py --data "D:/cta/data"
python -m unittest discover -s tests -v
```

`--data`目录包含原 `etf_daily.csv` 和 `etf_dividend.csv`，字段见DATA.md。默认评价2020—2025年，支持 `--start`、`--end`、`--out`。三只ETF使用相同日期与费用（每次手续费1‰、滑点1‰）。模型估计复用包内与输入序列哈希匹配的缓存，每窗1000次有效bootstrap；不重新估计、不减少原模型计算次数。

如果换了数据，先运行 `python run.py --data "D:/cta/data"` 生成匹配缓存，然后再执行改造程序；缺缓存时会明确报错，不会静默少算窗口。修改模型代码后应先用原程序的 `--rebuild-models` 重建缓存。原始数据不随工程再次分发。

## 输出和检验

`results/improvements/summary.csv`：3资产×8组，共24个结果，包含年化、夏普、卡玛、回撤、换手、平均绝对敞口、空仓天数比例与趋势日比例。

`yearly.csv`：所有年份；`*_signals.csv`：完整状态与信号；`*_nav.csv`、`*_fills.csv`、`*_trades.csv`：逐日账本、成交和平仓明细；`checks.csv`：净值独立重算及原基线逐日一致性；`manifest.json`：代码与缓存哈希、费用和日期。

本轮测试涵盖：不确定时不交易、ρ门槛移除、确认时序、下降趋势不做空和未来数据不改变历史；同时继续运行原15项测试。程序还校验每资产各方案日期相同、仅多头方案无负库存、现金+份额市值+应收等于净值，以及原baseline_full与包内旧净值逐日相同。

## 如何判断改善

按上表逐项比较，优先看费用后卡玛、回撤持续时间、换手与持仓比例，不能只看年化。少持仓自然可能少回撤，后续需补相近风险/现金暴露对照。2020—2025年已反复查看，本轮是回顾性机制实验，不是独立未见样本；不要逐年选择最佳版本再拼净值。

本轮范围是状态判断和入场改造。60/120日简单价格趋势、单独的震荡证据、退出确认/缓冲、波动率仓位管理、组合配置与ES作为后续阶段，尚未混入本程序。README中的设想不等于已经验证有效。

---

## 本轮验证与结果入口

已实际运行三资产×8组，共24个结果，19项测试通过；3个原完整基线逐日一致。结果见 `results/improvements/RESULTS.md` 和 `summary.csv`。黄金部分改造改善，但国债仍亏损，未得到跨资产一致有效的改造结论。没有将旧结果覆盖成新结果。

## 原工程说明（保留）

# 华泰时序CTA：研报规则的日频复现与差异审计

复现2026-06-10华泰证券《时序CTA方法论综述：市场状态、开仓信号与退出机制》的三资产单资产实验。

**状态：公开规则已经实现并实际运行，但未成功复现原报告的业绩。** 本地回测为2020—2025年，原报告到2026年5月；采用日频固定止损撮合，非分钟逐笔复原。未披露的规则全部列于 [RULES.md](RULES.md)。不包含宏观、配置或ES增量。

## 已包含

- 沪深300ETF 510300、黄金ETF 518880、国债ETF 511010，多空双向理想化账户。
- ER30自适应21—250日AR-GARCH、MAD/Huber清洗、1000次有效bootstrap、10日状态确认。
- 完整策略、恒趋势、恒震荡、无ATR（反向信号退出假设）和买入持有，共15个主结果。
- 无ATR字面删除、Wilder RSI、14日局部TR截断、无单向约束吊灯、零成本、成本翻倍，共18个敏感性结果。
- 次日开盘执行、ATR吊灯止损、跳空撮合、RSI40/60退出、每次1‰滑点+1‰手续费。
- 现金/库存/分红应收及支付账本、信号、成交、平仓交易、逐年指标、自包含HTML与核验记录。

## 运行

Python 3.10+。Windows可用conda；从头估计和模型测试需要可用的C++17编译器 `g++`（例如MSYS2 MinGW-w64，加入PATH）。Linux同样需要g++。

```bash
python -m pip install -r requirements.txt
python run.py --data /path/to/data
python verify_cache.py --data /path/to/data
python validate_results.py --data /path/to/data
python -m unittest discover -s tests -v
python build_report.py
```

`data`目录需放自己的 `etf_daily.csv` 和 `etf_dividend.csv`，格式见 [DATA.md](DATA.md)。路径可用Windows引号，例如 `--data "D:/cta/data"`。原始行情和研报PDF均不随GitHub包分发，不提供无授权再分发承诺。

提供已计算模型缓存。输入DataFrame哈希一致时复用8,848个窗口，不降低bootstrap次数；3个抽检窗口本次均独立重算、1000次有效结果与缓存完全一致。**这不是本轮重新完成8,848个估计。** 原始数据、行顺序或数值改变会使缓存失效，重新生成。

全量重新估计（耗时显著更长）：

```bash
python run.py --data /path/to/data --rebuild-models --workers 4
```

可用 `--start` 和 `--end` 指定测试日期，但不得把反复查看过的历史重新称作未见样本。无超参数选择，不按年挑最优。本项目是固定参数历史复现，不是训练/验证择优后真正独立的样本外检验。

## 结果入口

- `results/report.html`：打开即用，图片内嵌，无外部依赖。
- `results/summary.csv`：15个主结果；`yearly.csv`：逐年结果（包括字面无ATR对照）。
- `results/sensitivity.csv`：18个敏感性结果。
- `{code}_signals.csv`：所有预热与测试日的模型、CI、状态、指标。
- `{code}_{variant}_nav.csv / fills.csv / trades.csv`：日账本、所有成交、已平仓交易。
- `model_audit.csv / cache_recompute_audit.json / ledger_audit.json / manifest.json`：数据哈希、缓存核对、账本核对、环境与假设。
- `paper_reference.csv`：原文结果，区间不同，只供溯源，不作等区间绩效差值比较。

本实现的空头不含券源、融券利息、保证金与利息，不代表三只ETF可无条件实盘做空。现金与闲置分红利率设0。末日按市值记账，不人为平仓提高或降低交易胜率。胜率和盈亏比均来自费用后已平仓交易。

## 复现差异诊断（本轮新增）

```bash
python audit.py --data /path/to/data --optimizer-check
python build_audit_report.py
```

入口：`results/audit/audit_report.html`。包含状态条件通过率、实际净值盈亏归因、黄金主要亏损交易、相同规则仅多头/仅空头对照，以及15个窗口的替代求解器抽查。完整多空基线逐日重放与原交付一致，未更改阈值或交易规则。方向对照只用于归因，不声称为研报策略或已验证的增量。

主要发现：黄金趋势日仅3.78%，震荡空头是主要亏损来源；但仅多头年化0.35%，仍未跟上买入持有19.18%。国债零成本年化约−0.10%，研报费用后−5.49%。一个沪深300抽样窗口存在更优似然解，数值稳健性仍待扩大核查，未据此擅自改写全部模型缓存。当前结论仍为未成功复现原文业绩。
