"""Product entry point for the time-driven paper account."""
import streamlit as st

from src.extensions.position_agent.agent_paper import CAPITAL, PaperStore, default_db
from src.web.agent_panel import identity, receipt_panel, render_card


def render_paper():
    identity('510300 · 实时模拟账户',
             '北京时间交易日收盘后收取站内消息，确认的建议在下一交易日开盘模拟处理。')
    store = PaperStore(default_db())
    token = st.session_state.get('paper-agent-token')
    if token is None:
        create, restore = st.columns(2)
        with create:
            st.markdown('**开始模拟**')
            st.write('从 10 万元模拟现金开始。')
            if st.button('创建实时模拟账户', type='primary'):
                token = st.session_state['paper-agent-token'] = store.create()
                st.session_state['new-paper-recovery-code'] = True
                st.rerun()
        with restore:
            with st.form('restore-paper'):
                entered = st.text_input('恢复码', type='password')
                if st.form_submit_button('恢复实时模拟账户'):
                    try:
                        store.view(entered.strip())
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state['paper-agent-token'] = entered.strip()
                        st.rerun()
        return
    if st.session_state.get('new-paper-recovery-code'):
        with st.expander('请保存账户恢复码', expanded=True):
            st.code(token)
            if st.button('已保存恢复码'):
                st.session_state.pop('new-paper-recovery-code')
                st.rerun()
    if st.sidebar.button('退出当前实时模拟账户'):
        st.session_state.pop('paper-agent-token', None)
        st.session_state.pop('new-paper-recovery-code', None)
        st.rerun()
    _account(store, token)


@st.fragment(run_every='30s')
def _account(store, token):
    state, card, receipt = store.view(token)
    cols = st.columns(3)
    cols[0].metric('最近收盘净值', f"{state['equity']:,.2f} 元", f"{state['equity']/CAPITAL-1:+.2%}")
    cols[1].metric('可用现金', f"{state['account']['cash']:,.2f} 元")
    cols[2].metric('当前持有', f"{state['account']['quantity']:,} 份")
    if state['date']:
        st.caption(f"最近收盘结算日 {state['date']}。开盘成交后，净值于当天收盘更新。")
    receipt_panel(receipt)
    if card is None:
        st.info('等待实时模拟作业收到并核对首个交易日的收盘行情。')
        return

    def choose(current, target=None, *, adopt=False):
        store.choose(token, current, target, adopt=adopt)

    render_card(card, choose, lambda current: store.choose(token, current, decline=True), historical=False)
    with st.expander('最近的站内消息'):
        st.dataframe(store.inbox(token), hide_index=True)
