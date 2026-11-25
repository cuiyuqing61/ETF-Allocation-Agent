"""Offline import and integrity validation of reviewed corporate-action evidence."""
from hashlib import sha256
from pathlib import Path
import json
from src.trading.corporate_actions import validate_actions


def read_manifest(path):
    manifest_path = Path(path).resolve()
    raw=manifest_path.read_bytes();m=json.loads(raw)
    for evidence in m['sources'].values():
        evidence_path = manifest_path.parent / evidence['path']
        if sha256(evidence_path.read_bytes()).hexdigest()!=evidence['sha256']:
            raise ValueError('Corporate action source hash mismatch')
    validate_actions(m['actions'])
    if m['instrument']!='510300' or m['share_changes_in_coverage']!=[]:
        raise ValueError('Unsupported universe or share conversion')
    return sha256(raw).hexdigest(),m


def import_actions(store,path):
    version,m=read_manifest(path)
    # This verifies that actions attach to the exact existing raw-price snapshot.
    metadata,bars=store.load_position_dataset(m['dataset_id'])
    if m['coverage_start']!=metadata['start_date'] or m['coverage_end']!=metadata['end_date']:
        raise ValueError('Corporate action coverage differs from dataset')
    dates=set(bars.trade_date)
    for a in m['actions']:
        if not m['coverage_start']<=a['record_date']<=a['ex_date']<=m['coverage_end']:
            raise ValueError('Action outside certified coverage')
        if a['record_date'] not in dates or a['ex_date'] not in dates: raise ValueError('Action session missing')
    with store.connect() as c:
        c.execute('BEGIN IMMEDIATE')
        existing=c.execute('SELECT manifest_json FROM corporate_action_sets WHERE version=?',(version,)).fetchone()
        if existing:
            if json.loads(existing[0])!=m: raise ValueError('Action version conflict')
            return version,False
        c.execute('INSERT INTO corporate_action_sets VALUES(?,?,?,?)',(version,m['dataset_id'],json.dumps(m,sort_keys=True),str(Path(path).resolve())))
        for a in m['actions']:
            row=(m['dataset_id'],a['action_id'],m['instrument'],a['ex_date'],a['record_date'],a['pay_date'],a['cash_per_unit'],a['split_ratio'],a['source'])
            old=c.execute('SELECT * FROM corporate_actions WHERE dataset_id=? AND action_id=?',row[:2]).fetchone()
            if old and tuple(old)!=row: raise ValueError('Corporate action conflict')
            if not old:c.execute('INSERT INTO corporate_actions VALUES(?,?,?,?,?,?,?,?,?)',row)
    return version,True


def load_actions(store,version,dataset_id,start,end):
    with store.connect() as c:
        row=c.execute('SELECT manifest_path,manifest_json,dataset_id FROM corporate_action_sets WHERE version=?',(version,)).fetchone()
        if not row: raise ValueError('Import the frozen corporate-action manifest first')
        digest,m=read_manifest(row[0])
        if digest!=version or json.loads(row[1])!=m or row[2]!=dataset_id:
            raise ValueError('Corporate action version/dataset mismatch')
        if start<m['coverage_start'] or end>m['coverage_end']: raise ValueError('Unverified action coverage')
        for a in m['actions']:
            expected=(dataset_id,a['action_id'],m['instrument'],a['ex_date'],a['record_date'],a['pay_date'],a['cash_per_unit'],a['split_ratio'],a['source'])
            saved=c.execute('SELECT * FROM corporate_actions WHERE dataset_id=? AND action_id=?',expected[:2]).fetchone()
            if tuple(saved or ())!=expected: raise ValueError('Corporate action database mismatch')
    return m['actions']
