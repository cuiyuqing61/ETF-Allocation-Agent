"""Shared Agent cards, conversation and ledger receipts."""
import json
from pathlib import Path

import streamlit as st

from src.extensions.position_agent.agent_assistant import (
    AIHUBMIX_BASE_URL, ModelConfig, action_label, explain, model_connection_error,
    model_from_env, model_provider, respond,
)
from src.web.presentation import execution_text

ICON = Path(__file__).resolve().parent / 'assets/position-agent-icons/v3/01-matte-coin-robot-rounded.png'


def identity(title, description):
    image, body = st.columns([1, 9], vertical_alignment='center')
    with image:
        st.image(str(ICON), width=76)
    with body:
        st.title(title)
        st.caption(description)


def receipt_panel(receipt):
    if receipt is None:
        return
    with st.container(border=True):
        st.markdown('**最近成交回执**')
        reasons = {
            'opening_quote_unavailable': '开盘时段未收到可用行情，保留原份额。',
            'expired': '建议到达执行时点仍未确认，保留原份额。',
        }
        if receipt['reason'] in reasons:
            st.write(f"{receipt['execution_date']} · {reasons[receipt['reason']]}")
        elif receipt['reason'] == 'target':
            qty = receipt['signed_quantity']
            st.write(f"{receipt['execution_date']} · 模拟{'买入' if qty > 0 else '卖出'} {abs(qty):,} 份"
                     f" · 成交价 {receipt['fill_price']:.3f} 元 · 手续费 {receipt['fee_cash']:.2f} 元")
        else:
            st.write(execution_text(receipt))
        target = receipt.get('target_weight', receipt.get('target'))
        if receipt['reason'] != 'expired':
            st.caption('确认操作为保持份额。' if target is None else f'确认的目标仓位为 {target:.2%}。')
        if receipt.get('exposure') is not None:
            st.caption(f"当日收盘实际仓位 {receipt['exposure']:.1%} · 持有 {receipt['quantity']:,} 份"
                       f" · 账户净值 {receipt['equity']:,.2f} 元")
        else:
            st.caption('开盘回执已保存，收盘后更新实际仓位和账户净值。')


def _action(callback):
    try:
        callback()
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        return
    st.rerun()


def render_card(card, confirm, decline, *, historical):
    facts = card['facts']
    key = card['id']
    target = facts['target']
    with st.container(border=True):
        st.markdown(f"**站内消息 · {facts['date']} 收盘**")
        st.subheader(action_label(facts))
        current, proposed = st.columns(2)
        current.metric('当前仓位', f"{facts['exposure']:.1%}")
        proposed.metric('建议动作' if target is None else '目标仓位',
                        '保持份额' if target is None else f'{target:.1%}')
        st.write(explain(facts, use_llm=True))
        st.caption(f"计划在 {facts['execution_date']} 开盘模拟执行")
        with st.expander('查看依据'):
            st.write(f"收盘价 {facts['close']:.3f} 元 · 50 日均线 {facts['ma50']:.3f} 元")
            st.write(f"模型测算目标 {facts['mapped']:.2%} · 调仓阈值 {facts['band']*100:g} 个百分点")
            st.caption(f"行情截至 {facts['data_as_of']} · {facts['version']}")
        if card['status'] == 'complete':
            st.caption('这条建议已处理，可查看成交回执。')
            return
        if card['request']:
            request = card['request']
            if isinstance(request, str):
                request = json.loads(request)
            if historical:
                st.warning('操作已确认，结算尚未完成。')
                if st.button('恢复处理', key='resume-'+key):
                    _action(lambda: confirm(card, request['target'], adopt=request['adopt']))
            else:
                st.success('操作已确认，等待计划执行日开盘。')
            return
        if not card.get('actionable', True):
            st.info('这条卡片已到执行时点，请查看回执或等待下一条建议。')
            return
        if card['declined']:
            st.info('已拒绝原建议。可以在下方对话中自选仓位。')
        else:
            yes, no = st.columns(2)
            if yes.button('采纳并进入下一交易日' if historical else '确认按建议执行',
                          key='adopt-'+key, type='primary'):
                _action(lambda: confirm(card, adopt=True))
            if no.button('不采纳，和 Agent 聊聊', key='decline-'+key):
                _action(lambda: decline(card))
        if historical and st.button(
            '保持份额并进入下一交易日',
            key='hold-next-'+key,
            help='保留目前持有的 ETF 份额，完成下一交易日的模拟结算。',
        ):
            _action(lambda: confirm(card, None, adopt=False))
    _conversation(card, confirm, historical=historical)


def _chat_model():
    session_key = 'agent-chat-model-config'
    session_config = st.session_state.get(session_key)
    config = session_config or model_from_env()
    if config:
        st.caption(f'模型问答已配置（{config.provider}）。发送问题时会把最近对话和这张卡片的可见事实提供给模型服务。')
        if session_config and st.button('断开 AIHubmix', key='agent-aihubmix-disconnect'):
            del st.session_state[session_key]
            st.rerun()
        return config

    if model_provider() is None:
        st.caption('服务端接口地址不受支持，请核对 OPENAI_BASE_URL。')
    st.caption('当前使用本地问答。可以在下方连接 AIHubmix。')
    with st.expander('模型设置'):
        st.caption('密钥仅在当前页面会话使用，关闭或刷新页面后需重新连接。')
        with st.form('agent-aihubmix-settings'):
            api_key = st.text_input('AIHubmix API 密钥', type='password').strip()
            model_name = st.text_input('模型名', value='gpt-4o-mini').strip()
            connect = st.form_submit_button('验证并连接 AIHubmix')
        if connect:
            if not api_key or not model_name:
                st.error('请填写 AIHubmix API 密钥和模型名。')
            else:
                config = ModelConfig(AIHUBMIX_BASE_URL, model_name, api_key)
                with st.spinner('正在验证 AIHubmix 连接…'):
                    error = model_connection_error(config)
                if error:
                    st.error(error)
                else:
                    st.session_state[session_key] = config
                    st.rerun()


def _conversation(card, confirm, *, historical):
    facts = card['facts']
    key = card['id']
    memory = 'agent-chat-'+key
    if st.session_state.get('active-agent-chat') != memory:
        old = st.session_state.get('active-agent-chat')
        if old:
            st.session_state.pop(old, None)
        st.session_state['active-agent-chat'] = memory
    state = st.session_state.setdefault(memory, {'messages': [], 'draft': None})
    st.markdown('**问 Agent 或自选仓位**')
    config = _chat_model()
    question = st.chat_input('例如「回撤是什么意思」或「把仓位调到 35%」', key='chat-'+key)
    if question:
        reply = respond(question, facts, history=state['messages'], use_llm=True, model_config=config)
        state['messages'].extend([('user', question), ('assistant', reply.text)])
        state['messages'] = state['messages'][-10:]
        state['reply_source'] = reply.source
        state['reply_detail'] = reply.detail
        if reply.draft:
            state['draft'] = reply
    for role, message in state['messages']:
        with st.chat_message(role, avatar=str(ICON) if role == 'assistant' else None):
            st.text(message)
    if state.get('reply_source') == 'model':
        st.caption('本次由模型回答。')
    elif state.get('reply_source') == 'fallback':
        detail = state.get('reply_detail') or '本次模型回复不可用。'
        st.caption(f'{detail} 已显示本地答复。')
    draft = state['draft']
    if draft is None:
        return
    if draft.target is None:
        label = '保持原份额'
    else:
        side = '加仓' if draft.target > facts['exposure'] else '减仓' if draft.target < facts['exposure'] else '调整'
        label = f'{side}至 {draft.target:.2%}'
    with st.container(border=True):
        st.markdown('**待确认的自选操作**')
        st.write(f"当前仓位 {facts['exposure']:.1%} · {label} · {facts['execution_date']} 开盘模拟处理")
        accept, cancel = st.columns(2)
        confirmation = '确认自选并进入下一交易日' if historical else '确认自选操作'
        if accept.button(confirmation, key='manual-'+key, type='primary'):
            _action(lambda: confirm(card, draft.target, adopt=False))
        if cancel.button('取消草稿', key='cancel-'+key):
            state['draft'] = None
            st.rerun()
