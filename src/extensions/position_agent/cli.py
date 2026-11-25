"""Explicit research CLI; no network in validate/backtest, no broker execution."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import pandas as pd
import yaml
from src.data.data_store import DataStore
from src.extensions.position_agent.model_versions import ROOT, UPSTREAM_COMMIT



def config_bundle(path):
    path=Path(path).resolve()
    project=yaml.safe_load(path.read_text())
    rules=yaml.safe_load(path.with_name('market_rules.yaml').read_text())
    costs=yaml.safe_load(path.with_name('costs.yaml').read_text())
    return project,rules,costs


def code_fingerprint():
    paths=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'src/data').glob('*.sql'))+sorted(ROOT.glob('requirements-position-agent*.in'))+sorted(ROOT.glob('requirements-position-agent*.lock'))
    return sha256(b''.join(str(p.relative_to(ROOT)).encode()+p.read_bytes() for p in paths)).hexdigest()


def run_backtest(store,dataset_id,strategy,split,project,rules,costs):
    from src.strategies.etf_baselines import ETFBaselineStrategy
    from src.backtest.position_replay import ReplayConfig, replay
    if split=='test' and not project['final_test_enabled']:
        raise ValueError('Final test is frozen and disabled; preregistration required before consumption')
    metadata,bars=store.load_position_dataset(dataset_id)
    if metadata['instrument']!=project['instrument'] or metadata['adjustment_policy']!='none':
        raise ValueError('Dataset differs from frozen universe/adjustment')
    start,end=project['splits'][split]
    from src.data.corporate_actions import load_actions
    version=project.get('corporate_actions_version','unavailable')
    actions=load_actions(store,version,dataset_id,start,end) if version!='unavailable' else []
    return_basis='cash_dividends_accrual' if version!='unavailable' else 'price_only_excluding_dividends'
    from src.data.trading_calendar import get_trading_days_set
    required=get_trading_days_set(start,end,'SSE')
    if not required.issubset(set(bars.trade_date)):
        raise ValueError('Dataset does not cover the requested split')
    prior=bars[bars.trade_date<start]
    if len(prior)<20:
        raise ValueError('Need at least 20 prior sessions for warmup')
    selected=bars[(bars.trade_date>=prior.trade_date.iloc[-1]) & (bars.trade_date<=end)].copy()
    adapter=ETFBaselineStrategy(strategy)
    if strategy=='buy_hold':
        signals=adapter.generate_weights({'bars':selected,'execution':{'initial_capital':project['initial_capital'],'market_rules':rules,'costs':costs},'corporate_actions':actions}).weights
    else:
        signals=adapter.generate_weights({'bars':bars[bars.trade_date<=end]}).weights
        signals=signals.loc[pd.to_datetime(selected.trade_date.iloc[0]):]
    config=dict(project=project,market_rules=rules,costs=costs,strategy=strategy,split=split,
                dataset_id=dataset_id,code_hash=code_fingerprint(),return_basis=return_basis,corporate_actions_version=version)
    config_json=json.dumps(config,sort_keys=True)
    run_id=sha256(config_json.encode()).hexdigest()
    with store.connect() as conn:
        if conn.execute('SELECT 1 FROM experiment_runs WHERE run_id=?',(run_id,)).fetchone():
            return {'run_id':run_id,'created':False}
    replay_config = ReplayConfig(start, end, project['initial_capital'], rules, costs, actions)
    result = replay(replay_config, strategy, selected, signals)
    for d in result.decisions:
        d.update(run_id=run_id,decision_id=sha256(f"{run_id}:{d['decision_date']}".encode()).hexdigest())
    for a in result.daily_account: a['run_id']=run_id
    executions=result.trades.to_dict('records')
    # Explicit ids on the persisted copy; DataFrame itself remains a result artifact.
    for d,e in zip(result.decisions,executions):
        e.update(run_id=run_id,decision_id=d['decision_id'],execution_id=d['decision_id'])
        if pd.isna(e['fill_price']): e['fill_price']=None
    run=dict(run_id=run_id,strategy_version=strategy+'-v1',dataset_id=dataset_id,seed=None,split=split,
             evaluation_mode='historical',period_start=result.daily_account[0]['trade_date'],
             period_end=result.daily_account[-1]['trade_date'],initial_capital=project['initial_capital'],
             config_json=config_json,code_commit=UPSTREAM_COMMIT,status='completed_dividend_accounting' if actions else 'completed_price_only')
    created=store.save_position_run(run,result.decisions,executions,result.daily_account,result.corporate_action_events)
    return {'run_id':run_id,'created':created,'metrics':result.metrics,'return_basis':return_basis}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/position_agent/project.yaml'))
    p.add_argument('--base-dir')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('ingest')
    s=sub.add_parser('import-actions');s.add_argument('--manifest',required=True)
    s=sub.add_parser('reconcile');s.add_argument('--run-id',required=True)
    for name in ('validate-data','backtest','report'):
        s=sub.add_parser(name)
        if name!='report': s.add_argument('--dataset-id',required=True)
        if name=='backtest':
            s.add_argument('--strategy',choices=['cash','buy_hold','ma10_20'],required=True)
            s.add_argument('--split',choices=['train','validation','test'],default='validation')
            s.add_argument('--actions-version',default='unavailable')
    args=p.parse_args()
    project,rules,costs=config_bundle(args.config)
    store=DataStore(args.base_dir or str(ROOT/project['base_dir']))
    if args.command=='ingest':
        from src.data.akshare_provider import AkshareDataProvider
        provider=AkshareDataProvider(store.base_dir)
        metadata,bars=provider.fetch_snapshot(project['instrument'],project['start'],project['end'])
        dataset_id=store.save_position_dataset(metadata,bars)
        report=ROOT/'reports/position_agent'/f'data-quality-{dataset_id}.json'
        report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
        output={'dataset_id':dataset_id,'quality':metadata['quality']}
    elif args.command=='import-actions':
        from src.data.corporate_actions import import_actions
        version,created=import_actions(store,args.manifest)
        output={'corporate_actions_version':version,'created':created}
    elif args.command=='reconcile':
        from src.backtest.position_replay import reconcile
        with store.connect() as conn:
            capital=conn.execute('SELECT initial_capital FROM experiment_runs WHERE run_id=?',(args.run_id,)).fetchone()
            if capital is None: raise ValueError('Unknown run')
            daily=pd.read_sql_query('SELECT * FROM daily_account WHERE run_id=? ORDER BY trade_date',conn,params=(args.run_id,))
        if daily.prev_cash.isna().any(): raise ValueError('Legacy price-only run lacks v2 reconciliation fields')
        output={'run_id':args.run_id,**reconcile(daily.to_dict('records'),capital[0])}
    elif args.command=='validate-data':
        metadata,bars=store.load_position_dataset(args.dataset_id)
        output={'dataset_id':args.dataset_id,'rows':len(bars),'quality':metadata['quality']}
    elif args.command=='backtest':
        project['corporate_actions_version']=args.actions_version
        output=run_backtest(store,args.dataset_id,args.strategy,args.split,project,rules,costs)
    else:
        from src.data.position_store import METRICS_SQL
        with store.connect() as conn: output=pd.read_sql_query(METRICS_SQL,conn).to_dict('records')
    print(json.dumps(output,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
