"""Public historical replay with a per-session simulated account."""

import sqlite3

import plotly.graph_objects as go
import streamlit as st

from src.extensions.position_agent.demo import DEMO_END, DEMO_START, DemoTrial, default_db
from src.web.agent_panel import identity, receipt_panel, render_card


def _price_chart(bars):
    chart = go.Figure(go.Candlestick(
        x=bars.trade_date, open=bars.open, high=bars.high, low=bars.low, close=bars.close,
        increasing_line_color='#d64c4c', decreasing_line_color='#219c76', name='合成 K 线',
    ))
    chart.add_scatter(x=bars.trade_date, y=bars.close.rolling(50).mean(), name='50 日均线')
    chart.update_layout(height=340, xaxis_rangeslider_visible=False, yaxis_title='价格（元）',
                        margin=dict(l=8, r=8, t=12, b=8))
    st.plotly_chart(chart, width='stretch')


def render():
    identity('小金豆 · 历史模拟交易台', '用合成历史行情体验调仓建议、对话自选和模拟成交。')
    st.info('本页行情和账户收益为演示数据。模拟账户使用虚拟资金。', icon='ℹ️')
    session_key = 'etf-allocation-demo-session'
    current = st.session_state.get(session_key)
    if current is None:
        st.write(f'演示区间为 {DEMO_START} 至 {DEMO_END}，初始模拟资金 100,000 元。')
        if not st.button('开始历史模拟', type='primary'):
            return
    elif st.sidebar.button('新开一轮历史模拟'):
        current = None

    try:
        if current is None:
            path = default_db()
            trial = DemoTrial.start(path)
            st.session_state[session_key] = (trial.id, path)
            st.rerun()
        trial = DemoTrial(*current)
        card = trial.card()
        day, bars, daily, _ = trial.view()
        receipt = trial.receipt()
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        st.error(f'无法加载模拟账户。{exc}')
        return

    st.subheader(f'模拟时间 {day} 收盘')
    st.caption(f'进度 {trial.nodes.index(day)}/{len(trial.nodes) - 1} · 收盘出建议，下一模拟交易日开盘执行')
    user_rows = daily.loc[daily.run_id == f'{trial.id}-user']
    if user_rows.empty:
        equity, exposure, fees = trial.accounts['user'].cfg['capital'], 0., 0.
    else:
        latest = user_rows.iloc[-1]
        equity, exposure, fees = latest.equity, latest.exposure, user_rows.fee_cash.sum()
    assets, position, cost = st.columns(3)
    assets.metric('模拟账户净值', f'{equity:,.2f} 元')
    position.metric('实际仓位', f'{exposure:.1%}')
    cost.metric('累计手续费', f'{fees:,.2f} 元')

    left, right = st.columns([3, 2])
    with left:
        _price_chart(bars)
    with right:
        receipt_panel(receipt)
        if card:
            render_card(card, trial.confirm, trial.decline, historical=True)
        else:
            st.success('本轮历史模拟已完成。')
