"""Streamlit entry point for 小金豆."""

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web.agent_panel import ICON
from src.web.historical_panel import render as render_historical
from src.web.paper_agent_panel import render_paper


def main():
    st.set_page_config(page_title='小金豆 · ETF 调仓助手', page_icon=str(ICON), layout='wide')
    workspace = st.sidebar.radio('工作区', ['历史模拟交易台', '实时模拟账户'])
    if workspace == '历史模拟交易台':
        render_historical()
    else:
        render_paper()


if __name__ == '__main__':
    main()
