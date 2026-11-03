from pathlib import Path
import base64,html
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;P=ROOT/'results'/'audit'
NAMES={'510300':'沪深300ETF','518880':'黄金ETF','511010':'国债ETF'}

def tab(df,percent=()):
    x=df.copy()
    for c in percent:x[c]=x[c].map(lambda y:f'{y:.2%}' if pd.notna(y) else '—')
    return '<div class="scroll">'+x.to_html(index=False,border=0,na_rep='—')+'</div>'

def main():
    gates=pd.read_csv(P/'state_gates.csv',dtype={'code':str});attr=pd.read_csv(P/'pnl_attribution.csv',dtype={'code':str});direc=pd.read_csv(P/'direction_counterfactuals.csv',dtype={'code':str});tr=pd.read_csv(P/'closed_trade_attribution.csv',dtype={'code':str},parse_dates=['entry_date','exit_date']);opt=pd.read_csv(P/'independent_optimizer.csv',dtype={'code':str})
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True)
    cols=['tsi_pass','rho_positive','risk_pass','all_daily','confirmed_trend'];labs=['TSI','rho > 0','risk','all 3','10-day']
    for ax,(_,r) in zip(axs,gates.iterrows()):
        vals=r[cols].astype(float).to_numpy()*100;ax.bar(labs,vals,color=['#88b7bd']*3+['#42788c','#d1784b']);ax.set_title(r.code);ax.set_ylim(0,110);ax.grid(axis='y',alpha=.2)
        for j,v in enumerate(vals):ax.text(j,v+2,f'{v:.1f}%',ha='center',fontsize=9)
    axs[0].set_ylabel('Share of test days (%)');fig.suptitle('Independent gate pass rates | all 3 = conjunction; 10-day = confirmed state');fig.tight_layout();fig.savefig(P/'state_gates.png',dpi=150);plt.close(fig)
    gold=tr[tr.code=='518880'].sort_values('return_net').head(4).copy();s=pd.read_csv(ROOT/'results/518880_signals.csv',parse_dates=['date']);sx=s.set_index('date')
    for i,r in gold.iterrows():
        signal=s[s.date<r.entry_date].iloc[-1]
        gold.loc[i,'signal_date']=signal.date.strftime('%Y-%m-%d');gold.loc[i,'signal_rho']=signal.rho;gold.loc[i,'signal_rsi']=signal.rsi;gold.loc[i,'signal_state']=signal.state
    gold.to_csv(P/'gold_worst_shorts.csv',index=False)
    worst=gold.iloc[0];z=s[(s.date>='2025-02-20')&(s.date<='2025-05-15')]
    fig,axs=plt.subplots(2,1,figsize=(12,6),sharex=True,gridspec_kw={'height_ratios':[1.2,1]})
    axs[0].plot(z.date,z.adjclose/z.adjclose.iloc[0]*100,color='#007c83');axs[0].set_ylabel('Adjusted price (start=100)');axs[0].set_title('Gold short: 2025-03-19 to 2025-05-06 | net trade return -12.84%',loc='left')
    axs[1].plot(z.date,z.rsi,color='#8d5795',label='SMA RSI14');axs[1].axhline(70,color='#b44a41',ls='--',label='Short entry > 70');axs[1].axhline(60,color='#4c697f',ls='--',label='Short exit < 60');axs[1].set_ylabel('RSI');axs[1].legend(ncol=3,loc='best',fontsize=9)
    for ax in axs:ax.axvspan(worst.entry_date,worst.exit_date,color='#efc7c2',alpha=.4);ax.grid(alpha=.2)
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(P/'gold_short_example.png',dpi=150);plt.close(fig)
    def img(name):return '<img src="data:image/png;base64,'+base64.b64encode((P/name).read_bytes()).decode()+'" alt="研究诊断图">'
    g=gates[['code','rho_positive','all_daily','confirmed_trend','range']].rename(columns={'code':'资产','rho_positive':'ρ为正天数占比','all_daily':'当日全部条件通过','confirmed_trend':'确认趋势天数占比','range':'震荡天数占比'})
    a=attr.copy();a['code']=a.code.map(NAMES);a['direction']=a.direction.map({1:'多头',-1:'空头'});a['kind']=a.kind.map({'range':'震荡','trend':'趋势'});a['status']=a.status.map({'closed':'已平仓','open':'期末未平'})
    a=a[['code','kind','direction','status','trades','pnl_initial_capital']].rename(columns={'code':'资产','kind':'入场类型','direction':'方向','status':'状态','trades':'笔数','pnl_initial_capital':'对初始本金的盈亏贡献'})
    cf=direc[['code','strategy','cagr','max_drawdown','calmar']].copy();cf.code=cf.code.map(NAMES);cf.strategy=cf.strategy.map({'full':'多空基线','long_only':'只允许多头','short_only':'只允许空头'});cf.calmar=cf.calmar.map(lambda x:f'{x:.2f}');cf=cf.rename(columns={'code':'资产','strategy':'账户','cagr':'年化收益','max_drawdown':'最大回撤','calmar':'卡玛'})
    example=gold[['entry_date','exit_date','return_net','signal_date','signal_rho','signal_rsi']].copy();example.signal_rho=example.signal_rho.map(lambda v:f'{v:.4f}');example.signal_rsi=example.signal_rsi.map(lambda v:f'{v:.2f}');example=example.rename(columns={'entry_date':'开空日期','exit_date':'退出日期','return_net':'单笔费用后收益','signal_date':'信号日期','signal_rho':'信号日ρ','signal_rsi':'信号日RSI'})
    checks=pd.read_csv(ROOT/'results/sensitivity.csv',dtype={'code':str});cost=checks[checks.strategy=='zero_cost_diagnostic'][['code','cagr']].copy();base=pd.read_csv(ROOT/'results/summary.csv',dtype={'code':str});base=base[base.strategy=='full'][['code','cagr']];cost=cost.merge(base,on='code',suffixes=('_zero','_base'));cost.code=cost.code.map(NAMES);cost=cost.rename(columns={'code':'资产','cagr_zero':'零成本年化','cagr_base':'研报费用年化'})
    report='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CTA复现差异定位</title><style>body{margin:0;background:#f3f6f7;color:#203c4c;font:15px/1.85 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1140px;margin:auto;padding:30px}header{background:#153d4c;color:white;border-radius:15px;padding:28px}h1{font-size:30px;line-height:1.4}section{background:white;border-radius:12px;padding:26px;margin:22px 0}h2{font-size:22px}.note{border-left:4px solid #d79448;background:#fff5e8;padding:15px 20px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px 12px;border-bottom:1px solid #dde7eb;text-align:left;white-space:nowrap}th{background:#eaf2f5}img{width:100%;height:auto}details{padding:15px;background:#edf5f5}summary{cursor:pointer}pre{font:12px/1.6 monospace;white-space:pre-wrap}footer{color:#6b7c85;font-size:13px}@media(max-width:600px){main{padding:12px}section,header{padding:18px}h1{font-size:24px}}</style></head><body><main><header><p>REPLICATION AUDIT · 2026-11-01</p><h1>结果差在哪里：状态识别、黄金空头与国债成本</h1><p>基线保持不变 · 使用2020—2025年现有结果 · 诊断不等于优化成功</p></header>
<section><h2>核心发现</h2><div class="note">差异首先出现在市场状态识别，随后传导到交易方向。当前黄金只有3.78%的测试日被确认为趋势，96.22%为震荡；它在上涨阶段多次执行RSI反转做空。国债则同时面临微弱交易优势和高成本拖累。不能把这些问题仅归于分钟数据缺失。</div><p>本次未更改任何基线阈值、状态条件或交易参数。多空基线重放与上次交付逐日一致。归因按实际净值增量计算，闭合交易与期末未平交易的盈亏贡献之和与完整累计收益对齐，误差小于1e−10。</p></section>
<section><h2>1. 趋势识别被哪些条件挡住？</h2>'''+img('state_gates.png')+tab(g,['ρ为正天数占比','当日全部条件通过','确认趋势天数占比','震荡天数占比'])+'''<p>以黄金为例，TSI条件通过率73.81%，但ρ>0只在27.97%的日期成立；全部日条件通过率16.01%，要求连续10日后只剩3.78%的趋势日期。三个单独条件的通过率不是顺序筛选结果；图中“all 3”才是合取。</p><p>核对原文第15页黄金状态图，可见其存在持续较长的向上趋势区段，与本地状态路径明显不同。这里没有从图片估计原文精确趋势占比，也没有作者的逐日状态文件。当前证据只能定位差异所在，不能证明原文实际使用了哪种不同实现。</p><p><strong>关键含义：</strong>长期上涨不保证日收益率的一阶自相关系数为正。在目前对“收益率AR(1)”和“连续10日全部通过”的字面实现下，长期上涨也可能长期进入震荡策略。这是需要与作者程序核对的核心环节，不能擅自删除ρ条件或缩短10日确认后称为原文复现。</p></section>
<section><h2>2. 黄金亏损主要来自震荡空头，但问题不止做空</h2>'''+tab(a,['对初始本金的盈亏贡献'])+'''<p>黄金已平仓震荡空头33笔，对初始本金贡献−38.82个百分点，期末未平空头另为−0.88个百分点；震荡多头贡献+19.57个百分点，趋势多头贡献−12.47个百分点，总计约−32.61%。这是对初始本金的绝对盈亏分解，不是把单笔收益率简单相加，也不是各模块独立投资的收益。</p>'''+img('gold_short_example.png')+tab(example,['单笔费用后收益'])+'''<p>示例：2025-03-19开空，价格继续上涨，直到2025-05-06才按RSI退出，单笔费用后亏损12.84%。这些交易在信号日ρ均为负，因此进入震荡逻辑；原文震荡单用RSI退出，没有公开要求它们同时使用ATR。给所有震荡单加ATR也许会改变结果，但那将是新策略。</p></section>
<section><h2>3. 同规则方向对照：只做多也没有解决黄金问题</h2>'''+tab(cf,['年化收益','最大回撤'])+'''<p>三个账户使用相同状态、指标、费用、时期和账本。只做多/只做空仅屏蔽相反方向入场。移除一个方向会改变后续空仓时段和入场机会，因此这组反事实不能当作多空收益的加法分解。</p><p>沪深300只做多年化提高到6.23%，但卡玛约0.37，仍不足以证明稳定增量。黄金只做多年化只有0.35%，远低于同期买入持有19.18%；说明仅删除空头仍会错过上涨，状态识别和入场/退出路径必须继续核实。不能挑只做多版本取代研报多空结果。</p></section>
<section><h2>4. 国债成本：重要，但不是唯一问题</h2>'''+tab(cost,['零成本年化','研报费用年化'])+'''<p>国债零成本年化约−0.10%，计费后−5.49%；年化单边换手约27.72倍，一次往返滑点及手续费合计约4‰。不能简单用两条年化相减作为精确成本归因，因为成本会改变后续本金与仓位；但该对照明确显示交易优势不足以覆盖所设费用。黄金零成本仍年化−2.63%，其亏损也不是只靠降低费用就能消除。</p></section>
<section><h2>5. 优化器抽查：未发现黄金ρ方向被估反的证据</h2><p>对3资产×5个指定日期共15个窗口，以不同优化器（SciPy L-BFGS-B）和3组初始点重新优化同一AR-GARCH似然，另算OLS收益率AR(1)作为参照。黄金5个窗口的ρ均为负，替代优化器没有改变其符号。该检查复用同一似然函数，只是求解器交叉检查，不是完全独立的模型实现。</p><p>检查也发现局限：沪深300的2024-03-05窗口存在更低的负对数似然解（改善约0.171），ρ仍约0.06364。故不能宣布数值估计已被全面证明正确；该发现尚未通过全量模型及1000次bootstrap重算验证，不改写任何基线结果。15个抽样窗口也不能代表全部8,848个窗口。</p><details><summary>展开15个窗口的求解器记录</summary>'''+tab(opt)+'''</details></section>
<section><h2>下一步判断</h2><p>应优先核对作者的逐日状态序列、收益率定义与ρ估计方法、10日确认的状态保持/退出规则，然后核对入场是“价格在均线上”还是“刚刚上穿均线”，以及RSI平滑方法。当前代码采用研报文字的明确解释；这些未披露实现可能影响路径，但尚不能确定哪一项造成了原文差异。</p><p>在获得更明确实现证据或独立重建模型前，保留静态股债金等权作为研究基准，暂停宏观/ES增量。当前结论是定位了重要失效路径，尚未找到一个已证实的程序错误并修复，也尚未复现原报告盈利。</p></section><footer>原始来源：华泰证券2026-06-10《时序CTA方法论综述：市场状态、开仓信号与退出机制》第12—16页。原文区间至2026年5月，本地至2025年12月。代码包含audit.py、build_audit_report.py及全部诊断CSV；原始行情与研报PDF不再分发。</footer></main></body></html>'''
    (P/'audit_report.html').write_text(report,encoding='utf-8');print(P/'audit_report.html')
if __name__=='__main__':main()
