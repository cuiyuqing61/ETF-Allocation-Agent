<img src="src/web/assets/position-agent-icons/v3/01-matte-coin-robot-rounded.png" alt="小金豆金币机器人" width="128">

# ETF Allocation Agent

Daily position sizing and paper trading for the CSI 300 ETF (510300).

小金豆面向刚接触 ETF 的用户。它用沪深 300 ETF（510300）的最近 20 个交易日行情和当前账户状态生成仓位建议，并说明调仓依据。用户可以采纳建议、通过对话自选仓位，或保持原有份额。确认后，交易台在下一模拟交易日开盘处理操作并提供回执。

策略采用 PPO 训练日频 Transformer，结合 50 日均线和调仓阈值生成建议。账户使用虚拟资金，买卖按整手、手续费、滑点和分红规则记账。

## 快速开始

在项目根目录使用 Python 3.11 安装依赖并启动。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-position-agent-rl.lock
python -m streamlit run src/web/app.py --server.address 127.0.0.1
```

打开终端打印的本地地址，点击「开始历史模拟」。演示使用合成行情，首次开始模拟时会在 `data/demo/` 新建 SQLite 账本。你可以采纳一条建议，或输入「把仓位调到 35%」并确认，查看成交后的仓位和费用。

## 大模型问答

默认提供本地术语解释和对话自选仓位。需要大模型问答时，在「模型设置」中填入 AIHubmix 密钥并验证连接，默认模型为 `gpt-4o-mini`。密钥只用于当前页面会话，刷新后需重新连接。

也可以在启动 Streamlit 前，用环境变量 `OPENAI_API_KEY` 和 `OPENAI_MODEL` 配置服务端。OpenAI 使用默认接口。使用 AIHubmix 时，将 `OPENAI_BASE_URL` 设为 `https://aihubmix.com/v1`。

启用大模型后，服务会收到近期对话和当前卡片的可见事实。程序校验模型引用，接口异常时显示本地答复。模拟成交以用户在页面确认的指令为准。

## 实时模拟

侧边栏的「实时模拟账户」使用现实交易日。启动行情作业前，准备覆盖处理区间的 510300 分红及份额变动清单，并核对所引公告。页面和作业默认共用 `data/position_agent/agent-paper.sqlite3`，可在两端设置相同的 `POSITION_AGENT_PAPER_DB` 自定义路径。

```bash
python -m src.extensions.position_agent.agent_paper --actions data/reviewed-actions.json
```

行情作业默认使用新浪，可通过 `--feed eastmoney` 改用东方财富。历史演示可离线使用。实时模拟须获取当日行情和覆盖当日的已核对资料，仓位调整仍须在页面确认。

分红清单包含 `instrument`（`510300`）、`coverage_start`、`coverage_end`、`share_changes_in_coverage`、`actions` 和 `sources`。每条现金分红需要 `action_id`、`record_date`、`ex_date`、`pay_date`、`cash_per_unit`、`split_ratio`（`1`）和 `source`。`sources` 保存证据文件的 `path` 和 `sha256`。程序会核对证据哈希。训练用清单还需填入行情下载返回的 `dataset_id`，起止日期与该行情快照一致。

## 训练与验证

用自己的行情与公告文件运行。依次使用前两条命令返回的 `dataset_id` 和 `corporate_actions_version` 替换训练命令中的占位符。

```bash
python -m src.extensions.position_agent.cli ingest
python -m src.extensions.position_agent.cli import-actions --manifest data/reviewed-actions.json
python -m src.extensions.position_agent.train_daily_iteration \
  --dataset-id DATASET_ID --actions-version ACTIONS_VERSION \
  --output reports/training/seed11
```

训练期为 2015–2021 年，验证期为 2022–2024 年。命令在输出目录保存新权重、标准化参数、配置、校验清单及验证指标。三类基线可通过 `python -m src.extensions.position_agent.cli backtest --help` 查看参数。执行 `python -m pytest -q` 可离线验证历史模拟及实时模拟的记账流程。

## 2025 年独立回测

模型参数在回测前冻结，以 10 万元虚拟资金独立回放了 243 个交易日。两组使用相同的 50 日均线、调仓阈值和成交成本。

| 策略 | 净收益 | 最大回撤 |
| --- | ---: | ---: |
| Transformer（部署模型，种子 11） | 4.99% | 2.87% |
| 同规则固定半仓 | 6.28% | 3.45% |

回测记录对应真实的 510300 历史研究快照。仓库包含部署模型、校验清单及 [冻结协议](models/trend-sized-10bp/independent-2025/protocol.json)。公开演示的合成行情用于体验应用流程，回测复现需要自行获取研究所用的行情和分红资料。

## 许可

代码采用 [Apache-2.0](LICENSE) 许可。项目基于 FinRL-X 的部分组件，详见 [NOTICE](NOTICE)。
