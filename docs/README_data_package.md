# 数据包说明（tsc_cta_data_v3）

生成时间：2026-11-01 ｜ 覆盖窗口：2010-01-01 ~ 2025-12-31 ｜ 频率：日频

用于「沪深300ETF / 黄金ETF / 国债ETF 三资产 + 宏观指标」的日频股债金研究（CTA 择时、
配置、VaR/ES 均可直接用），可直接作为华泰时序 CTA 复现项目的输入层。

**v3 相对 v2 的三处修正**（针对评审意见）：
1. 分红校验重做——先验证 `LJJZ` 口径，再对全历史做台阶扫描，并与公告逐笔比对；
   旧报告里的「17 笔全部通过」已作废。
2. `available_time` 如实标注为**推算值**（`assumed_rule`），并新增频率、日期语义、
   保守滞后、失效阈值四列；周频指标另给保守版本 `macro_daily_strict.csv`。
3. 去掉所有硬编码绝对路径（`scripts/paths.py`），原始宏观表随包附上，补 `requirements.txt`。

---

## 一、包内结构

```
tsc_cta_data_v3_20261101/
├── README_PACKAGE.md              ← 本文件
├── requirements.txt               ← 依赖清单（pandas/numpy/openpyxl…）
├── data/
│   ├── etf_daily.csv              9,437 行  未复权 OHLCV + volume + adj_factor
│   ├── etf_total_return.csv       9,437 行  逐日总回报：ret_total / tr_index / close_adj
│   ├── etf_dividend.csv              21 条  分红送配（含 2026 年 7 条，备扩展）
│   ├── etf_dividend_events.csv       17 条  窗口内分红事件
│   ├── dividend_check_detail.csv          三源对照（表内 / F10 公告 / 净值）
│   ├── macro_daily.csv           58,487 行  15 项指标（含可用时间与频率标注）
│   ├── macro_daily_strict.csv    58,487 行  同上，**保守可用时间**版本
│   ├── macro_coverage.csv            15 行  覆盖度 + 频率 + 失效阈值
│   ├── index_daily.csv            6,432 行  000300 等指数（仅代理检验）
│   ├── DATA_DICTIONARY.md                  字段级口径（先看这个）
│   ├── SOURCE_LOG.md                       抓取来源与失败链路
│   ├── REVIEW_RESPONSE_2026-11-01.md       对两轮评审意见的逐条回应
│   ├── qc_report.md                        质检报告
│   ├── data_manifest.json                  条数/区间/sha256/缺口/校验汇总
│   └── raw/
│       ├── 高频宏观指标.xlsx               原始 Wind 导出表（复现必需，1.0 MB）
│       ├── nav_510300.csv 等 3 份            基金单位净值/累计净值快照（分红校验用）
│       └── f10_dividend_510300.csv 等 3 份   F10 分红送配快照
├── reports/
│   ├── dividend_verification_v3.md        ★ 分红完整性校验（v3，替代旧报告）
│   ├── etf_adjust_validation.md           复权因子三重校验
│   └── etf_dividend_independent_check.md  （已作废）v1 比值法，保留仅作对照
└── scripts/
    ├── paths.py                  路径解析（CLI / 环境变量 / data/raw，无硬编码）
    ├── fetch_backup.py           腾讯行情 + 天天基金分红 + 新浪COMEX
    ├── build_adjust.py           重建复权因子与总回报（含自检）
    ├── patch_macro.py            补 BDI/R007/1Y + 可用时间/频率/失效阈值
    ├── verify_dividend_full.py   ★ 分红四步校验（口径/全历史/公告/注入）
    ├── qc_check.py               生成 qc_report.md
    ├── make_manifest.py          生成 data_manifest.json
    ├── fetch_amount_bdi.py       东财解封后补 amount 与 BDI 尾巴
    └── check_independent.py      （已作废）v1 分红核对
```

---

## 二、环境与依赖

```bash
pip install -r requirements.txt
# 核心：pandas>=2.2  numpy>=1.26  openpyxl>=3.1（读原始 xlsx）
# 回测另需：scipy  matplotlib  hmmlearn（状态识别，非本包所必需）
```

Python ≥ 3.10（脚本用了 `X | None` 类型注解）。全部脚本只依赖公开免费数据源，
不需要 Wind 终端。

---

## 三、五分钟上手

```python
import pandas as pd

# 1) 总回报序列（回测直接用这个，不要用未复权价算收益）
tr = pd.read_csv("data/etf_total_return.csv", dtype={"code": str}, parse_dates=["date"])
px = tr.pivot(index="date", columns="code", values="close_adj")     # 后复权价
ret = tr.pivot(index="date", columns="code", values="ret_total")    # 日总回报

# 2) 三资产共同覆盖区间
common = px.dropna(how="any").index
print(common.min(), common.max(), len(common))   # 2013-07-29 ~ 2025-12-31，约 3,024 个交易日

# 3) 宏观指标：按 available_time 入池（严格检验换 strict 版）
mac = pd.read_csv("data/macro_daily.csv", parse_dates=["observation_date", "available_time"])
s = mac[mac["indicator"] == "中债国债到期收益率:10年"] \
        .set_index("available_time")["value"].sort_index()
```

**三条硬规则**

1. 收益一律用 `ret_total`（或 `close_adj` 差分），**不要**用 `close` 直接算——三只 ETF 都有现金分红。
2. 宏观一律按 `available_time` 对齐到交易日 `t`；`observation_date` 只用于展示。
3. **不要**把 `available_time` 当成真实发布时间。它是按发布规则推算的估计值，
   严格检验请用 `macro_daily_strict.csv`，并把结果视为上限（真实可用只会更晚）。

---

## 四、复权口径

```
r_t  = (P_t + d_t) / P_{t-1} - 1        # 除息日当天总回报
TR_t = ∏(1 + r_t)                        # 总回报指数，TR_0 = 1
F_t  = TR_t / (P_t / P_0)                # 后复权因子，首日 = 1
```

除息日因子跳变：`F_t / F_{t-1} = 1 + d_t / P_t`（分母是**除息日当天收盘**）。
校验见 `reports/etf_adjust_validation.md` 与 `reports/dividend_verification_v3.md` 第 6 节。

---

## 五、分红校验怎么做的（v3）

| 步骤 | 做什么 | 结果 |
|---|---|---|
| 1 口径判定 | 检验 `LJJZ = (DWJZ + C)/k` 的 k 是否恒定 | k 波动 ≤0.06% → 确认「加分红、不再投资」 |
| 2 全历史扫描 | 对 `(DWJZ+C) − k·LJJZ` 残差做台阶扫描，门槛 0.2% 净值/日 | 检出 5 个事件，**表外 0 个** |
| 3 公告核对 | F10 分红送配页逐笔比对日期与金额 | 21/21 一致 |
| 4 注入检验 | 逐笔删掉表内分红重扫 | 21/21 被检出 |
| 5 因子外部校验 | 净值口径 TR vs 价格口径 TR | 年化差 +2.4 / −3.1 / +2.4 bp |

**为什么 v1 的结论作废**：v1 用 `(1+LJJZ增长)/(1+DWJZ增长)−1` 当隐含分红率，
但 LJJZ 不含再投资，比值里混进了 `r·(1−P/L)` 的行情项；且只看表内 17 笔，
发现不了表外漏记。

**能力边界**：净值小数位决定可分辨下限（510300/518880 ≈0.0003 元/份；
511010 的 LJJZ 只有 3 位小数 → ≈0.10 元/份），低于该量级只能靠公告。
未取得基金公司公告原文（F10 为二手转录），发布级证据需按公告编号归档 PDF。

---

## 六、已知缺口（如实标注，未插补）

| 项目 | 状态 | 说明 |
|---|---|---|
| `amount` 成交额 | **全空** | 腾讯源不提供、东财 push2his 限流。**未用估算值填充**；容量研究前必须补实测值 |
| BDI / R007 / 1年国债 | 止于 2025-07-10 | 后 5.5 个月缺失；R007 不可用 Shibor3M 直接替换 |
| PTA/聚酯/江浙织机 | 止于 2025-04-17 | 上游 Wind 停更 |
| 建材、秦皇岛 | 止于 2025-07 | 上游停更 |
| `available_time` | 全部为估算 | `assumed_rule`，非真实发布时间 |
| 分钟线 | 无 | 复现研报的「分钟级监控规则」需另取，本包只到日频 |

三只 ETF 起点不同（510300 2012-05-28、511010 2013-03-25、518880 2013-07-29），
**共同可回测区间 2013-07-29 ~ 2025-12-31（约 12.4 年）**。

关于 2026 年数据：基金净值与分红快照已到 **2026-10-29**，说明行情可续抓；
本包按既定窗口截止 2025-12-31。如需延长窗口，改 `config` 里的 `END` 后重跑
`fetch_backup.py → build_adjust.py → patch_macro.py` 即可（分红表已含 2026 年 7 笔）。

---

## 七、复现顺序

```bash
python scripts/fetch_backup.py         # 1. 抓行情/分红/COMEX（原始表默认读 data/raw/）
python scripts/build_adjust.py         # 2. 建复权因子与总回报
python scripts/patch_macro.py --strict # 3. 补宏观 + 可用时间（另出 strict 版）
python scripts/verify_dividend_full.py # 4. 分红四步校验（离线复跑加 --offline）
python scripts/qc_check.py             # 5. 出质检报告
python scripts/make_manifest.py        # 6. 出清单与校验和
```

原始宏观表路径解析顺序：`--macro-xlsx` 参数 → 环境变量 `TSC_MACRO_XLSX` →
`data/raw/高频宏观指标.xlsx`。找不到会报错并列出查过的路径，**不会静默跳过**。

---

## 八、完整性校验

```bash
python scripts/make_manifest.py     # 重新计算 sha256
```

对照 `data/data_manifest.json` 的 `files` 与 `raw_snapshots` 两节，
即可确认包内文件未被改动。
