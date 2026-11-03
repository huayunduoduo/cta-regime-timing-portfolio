"""Build a standalone HTML from run_improvements.py outputs; no model fitting."""
from pathlib import Path
import argparse,base64,html
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'results/improvements');a=p.parse_args();r=a.results
 if not (r/'summary.csv').exists():raise SystemExit('Missing summary.csv. Run: python run_improvements.py --data data')
 s=pd.read_csv(r/'summary.csv',dtype={'code':str});y=pd.read_csv(r/'yearly.csv',dtype={'code':str});parts=[]
 labels={'baseline_full':'原多空基线','baseline_long':'原规则仅多头','neutral_10':'非趋势不交易','no_rho_10':'取消rho·10日','no_rho_3':'取消rho·3日','no_rho_3of5':'取消rho·5日内3日','risk_separate_3of5':'独立波动过滤','buyhold':'买入持有'}
 def table(df):
  df=df.copy()
  for c in ['cagr','max_drawdown','mean_exposure','cash_day_share','total_return']:
   if c in df:df[c]=df[c].map(lambda x:f'{x:.2%}')
  for c in ['calmar','sharpe']:
   if c in df:df[c]=df[c].map(lambda x:f'{x:.3f}')
  df['strategy']=df.strategy.map(labels)
  return '<div class="scroll">'+df.to_html(index=False,border=0)+'</div>'
 for code,name in [('510300','沪深300ETF'),('518880','黄金ETF'),('511010','国债ETF')]:
  fig,ax=plt.subplots(figsize=(12,5))
  for v in s[s.code==code].strategy:
   z=pd.read_csv(r/f'{code}_{v}_nav.csv',parse_dates=['date']);ax.plot(z.date,z.nav,label=v)
  ax.legend(fontsize=8,ncol=2);ax.set_title(code+' | Fixed state-rule experiments');ax.set_ylabel('NAV');ax.grid(alpha=.2);fig.tight_layout();path=r/f'{code}_improvements.png';fig.savefig(path,dpi=130);plt.close(fig)
  img=base64.b64encode(path.read_bytes()).decode();parts.append(f'<section><h2>{name}</h2><img src="data:image/png;base64,{img}">'+table(s[s.code==code][['strategy','cagr','max_drawdown','calmar','sharpe','mean_exposure','cash_day_share']])+'<details><summary>逐年结果</summary>'+table(y[y.code==code][['strategy','year','total_return','max_drawdown','calmar']])+'</details></section>')
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>状态改造实验结果</title><style>body{font:15px/1.8 system-ui,"Microsoft YaHei";margin:0;background:#f1f5f6;color:#203c49}main{max-width:1200px;margin:auto;padding:25px}section,header{padding:24px;background:white;border-radius:12px;margin:20px 0}header{background:#173e4b;color:white}img{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:left}details{margin-top:18px}summary{cursor:pointer}</style><main><header><h1>ETF状态识别改造：完整结果</h1><p>固定方案逐项对照；没有根据结果择优，不代表样本外增量已经成立。</p></header><section><p>字段：cagr为年化收益，max_drawdown为最大回撤，calmar为卡玛，sharpe为夏普；mean_exposure为平均绝对敞口，cash_day_share为空仓天数占比。各版本含义见README。原多空基线只作参考，与仅多头方案方向权限不同。</p><p>不确定状态不新开仓；已有持仓继续按原ATR退出。所有费用、分红与账本复用原引擎。历史结果已反复查看，不能称为全新未见样本。</p></section>'''+''.join(parts)+'</main></html>'
 out=r/'improvements_report.html';out.write_text(page,encoding='utf-8');print('HTML saved:',out.resolve())
if __name__=='__main__':main()
