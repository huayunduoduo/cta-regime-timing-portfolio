# 基于市场状态识别的股债金择时复现与多头组合研究

复现华泰证券 2026-06-10《时序CTA方法论综述：市场状态、开仓信号与退出机制》，并沿"复现 → 诊断 → 改进 → 配置"四步推进：原多空策略复现未达原报告业绩（如实记录），转为仅多头分资产改造，最终落地为低换手股债金 ETF 配置方案。全部结果可复现、失败结果全部保留。

## 研究路线（按顺序阅读）

| 阶段 | 文档 | 内容 |
|---|---|---|
| 1. 研报复现与差异审计 | [docs/README_1_replication_audit.md](docs/README_1_replication_audit.md) | ER30 自适应 21–250 日窗口、AR(1)-GARCH(1,1)、1000 次 Bootstrap、ATR 吊灯止损的日频复现；15 个主结果 + 18 个敏感性结果，差异逐项审计 |
| 2. 仅多头分资产策略 | [docs/README_2_long_only.md](docs/README_2_long_only.md) | 24 组资产×规则消融，分资产定制规则（国债 MA120 缓冲后年化 -4.53%→+2.75%、换手 30.4→0.86） |
| 3. 股债金配置研究 | [docs/README_3_allocation.md](docs/README_3_allocation.md) | 36 组配置候选对照（3 趋势 × 3 配置 × 2 调仓 × 2 ES），推荐月度逆波动配置 |
| — | [docs/README_data_package.md](docs/README_data_package.md) | 数据包 tsc_cta_data_v3 说明（分红校验重做、宏观数据口径） |
| — | [docs/README_windows_quickstart.md](docs/README_windows_quickstart.md) | Windows 小白快速运行指南 |

配套报告（HTML，仓库内直接打开）：

- `results/portfolio/portfolio_report.html` —— 股债金配置主报告（最终结论）
- `results/improvements/improvements_report.html` —— 状态改造实验
- `results/report.html` —— 研报复现报告
- `results/audit/audit_report.html` —— 复现差异审计
- `index.html` —— 报告导航页

## 主结果（2020-01-01 至 2025-12-31，佣金 1‰ + 滑点 1‰）

| 组合 | 年化收益 | 最大回撤 | 卡玛比 |
|---|---:|---:|---:|
| **月度逆波动配置（主方案）** | **7.50%** | **-4.83%** | **1.552** |
| 年度滚动前三组合 | 7.64% | -5.00% | 1.526 |
| 月度等权配置 | 9.14% | -11.69% | 0.782 |
| 仅多头分资产三子账户 | 6.45% | -5.48% | 1.18 |
| 原多空 CTA（各 1/3 初始资金） | -2.28% | -27.78% | -0.082 |

诚实声明：主方案 2022 年仍亏 0.58%，2020–2022 三年平均仅 3.31%；2020–2025 数据在此前研究中已被查看，**不是全新样本外**，冻结规则待新数据验证。

## 快速开始

Python 3.10+。阶段 3 入口不依赖 C++ 与缓存，数十秒出结果。**仓库不随包分发行情数据**（授权原因），需自备 `data/etf_daily.csv`、`data/etf_dividend.csv`，字段格式见 [DATA.md](DATA.md)：

```powershell
conda activate cta_research          # 或自建环境
python -m pip install -r requirements.txt
python portfolio_research.py --data data
```

成功后打开 `results/portfolio/portfolio_report.html`。阶段 1/2 的运行方式见各自 README（阶段 1 模型重估需要 g++ 编译器）。

追加新数据：向 `data/etf_daily.csv`、`data/etf_dividend.csv` 追加真实新行情（三 ETF 交易日对齐），然后：

```powershell
python portfolio_research.py --data data --end 2026-05-29 --out results/portfolio_20260529
```

## 仓库结构

```
├── portfolio_research.py      # 阶段3：股债金配置回测入口
├── run.py                     # 阶段1：研报复现入口
├── run_improvements.py        # 状态改造实验入口
├── engine.py / model.py       # 撮合引擎与 AR-GARCH 模型
├── docs/                      # 各阶段 README 与学习资料
├── data/                      # etf_daily.csv / etf_dividend.csv
├── results/                   # 全部结果（含失败结果）
├── tests/                     # 单元测试（26 项，会计误差 < 1e-12）
├── RULES.md                   # 未披露规则的完整清单
└── DATA.md                    # 数据格式说明
```
