"""Grounded explanations and conversational drafts for 510300 paper accounts."""
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
import json
import os
import re

import requests

from src.extensions.position_agent.daily import canonical

DEFAULT_BASE_URL = 'https://api.openai.com/v1'
AIHUBMIX_BASE_URL = 'https://aihubmix.com/v1'
MODEL_PROVIDERS = {
    DEFAULT_BASE_URL: ('OpenAI', 'max_completion_tokens'),
    AIHUBMIX_BASE_URL: ('AIHubmix', 'max_tokens'),
}
ACTION = {'exit': '目标仓位 0%', 'cash': '保持空仓', 'hold': '保持原份额', 'buy': '加仓', 'sell': '减仓'}
TERMS = {
    '仓位': '仓位是 ETF 市值占账户总资产的比例。仓位 40% 表示约四成资产持有 ETF。',
    '份额': '份额是持有 ETF 的数量。保持份额表示不主动买卖，仓位比例仍会随价格变化。',
    '均线': '50 日均线是截至当日最近 50 个交易日收盘价的平均值。第六套用收盘价和均线的位置关系决定是否允许持仓。',
    '回撤': '回撤是当前总资产较此前最高点的下降比例。10 万元跌到 9 万元，如果 10 万元是之前的最高点，回撤就是 10%。',
    'ETF': 'ETF 是可以在交易所买卖的基金。本交易台模拟的是 510300。',
    '百分点': '从 50% 调到 40% 是减 10 个百分点。把现有仓位按比例减少 10%，则从 50% 变为 45%。',
    '滑点': '滑点是模拟成交价与参考开盘价之间的不利差额，已计入成交价。',
    '手续费': '手续费是账户在成交时另行扣除的费用。实际扣费以成交回执为准。',
    '分红': '基金分红是基金向符合权益登记条件的持有人分配的现金。账本分别记录分红权益和到账现金。',
    '净值': '账户净值是现金、ETF 市值和已经确认但尚未收到的分红之和。',
    'K线': '一根日 K 线展示一个交易日的开盘、收盘、最高和最低价。',
    '整手': '本模拟按 100 份一手计算买入量，实际成交仓位会受价格、费用和份额取整影响。',
    '收益': '账户净收益由期末总资产与起始金额比较得到，账本会计入成交费用和分红。',
    '涨跌停': '价格到达当日涨跌幅边界时，本模拟会按现有规则处理相应的买卖，可能未成交。',
    '建仓': '建仓是从空仓开始买入，建立最初的持仓。比如原来持有 0 份 ETF，第一次买进 100 份就是建仓。',
    '空仓': '空仓表示账户当前持有 0 份 ETF。',
    '满仓': '满仓表示账户资产几乎都用于持仓，可用现金很少。',
}
TARGET = re.compile(
    r'(?:(?:请|帮我|我想|我要|请帮我)\s*)?(?:把)?(?:目标)?仓位\s*'
    r'(?:调整到|调到|调至|调成|设为|改为|改成|到|设成)\s*(\d{1,3}(?:\.\d{1,2})?)\s*%'
)


def action_label(facts):
    return ACTION[facts['reason']]


def explanation_parts(facts):
    relation = '高于' if facts['close'] > facts['ma50'] else '在或低于'
    market = f"收盘价 {facts['close']:.3f} 元{relation} 50 日均线 {facts['ma50']:.3f} 元。"
    if facts['reason'] in {'exit', 'cash'}:
        rule = '均线规则将目标仓位设为 0%。'
    else:
        difference = abs(facts['difference'])*100
        comparison = '未达' if facts['reason'] == 'hold' else '达到'
        outcome = '保留原份额' if facts['reason'] == 'hold' else '按目标调仓'
        shown = f'{difference:.2f}'
        if round(difference, 2) == facts['band']*100 and difference != facts['band']*100:
            shown = f'{difference:.6f}'.rstrip('0').rstrip('.')
        rule = (f"当前仓位 {facts['exposure']:.1%}，模型测算目标 {facts['mapped']:.1%}，"
                f"相差 {shown} 个百分点，{comparison} "
                f"{facts['band']*100:g} 个百分点的阈值，建议{outcome}。")
    return {'market': market, 'rule': rule}


def _base_url():
    return (os.getenv('OPENAI_BASE_URL', '').strip() or DEFAULT_BASE_URL).rstrip('/')


def model_provider():
    provider = MODEL_PROVIDERS.get(_base_url())
    return provider[0] if provider else None


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    model_name: str
    api_key: str = field(repr=False)

    @property
    def provider(self):
        return MODEL_PROVIDERS[self.base_url][0]


def model_from_env():
    base_url = _base_url()
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    model_name = os.getenv('OPENAI_MODEL', '').strip()
    if base_url in MODEL_PROVIDERS and api_key and model_name:
        return ModelConfig(base_url, model_name, api_key)


class _ModelError(Exception):
    """A user-facing failure message that never includes request or response contents."""


def _http_failure(response, provider):
    if response is None:
        return f'{provider} 接口请求失败。'
    status = response.status_code
    try:
        code = response.json()['error']['code']
    except (ValueError, KeyError, TypeError):
        code = None
    if status == 401:
        return f'{provider} 拒绝了当前 API 密钥（401）。'
    if status == 403:
        return (f'{provider} API 在当前地区不可用（403）。' if code == 'unsupported_country_region_territory'
                else f'{provider} 拒绝了请求（403），请检查密钥权限。')
    if status == 404:
        return f'{provider} 未找到所选模型（404），请核对模型名和密钥权限。'
    if status == 402:
        return f'{provider} 拒绝了请求（402），请检查账户余额和额度。'
    if status == 429:
        return (f'{provider} API 额度不足（429），请检查账户余额和额度。'
                if code in ('credit_balance_exhausted', 'insufficient_quota', 'billing_hard_limit_reached')
                else f'{provider} 请求太频繁（429），请稍后再试。')
    if status >= 500:
        return f'{provider} 服务暂时出错（{status}），请稍后再试。'
    return f'{provider} 拒绝了本次请求（{status}）。'


def _request_json(config, messages, limit, *, read_timeout=8):
    provider, token_field = MODEL_PROVIDERS[config.base_url]
    try:
        response = requests.post(
            f'{config.base_url}/chat/completions',
            headers={'Authorization': 'Bearer ' + config.api_key},
            json={'model': config.model_name, 'response_format': {'type': 'json_object'},
                  token_field: limit, 'messages': messages},
            timeout=(3, read_timeout), allow_redirects=False,
        )
        response.raise_for_status()
        result = json.loads(response.json()['choices'][0]['message']['content'])
    except requests.HTTPError as exc:
        raise _ModelError(_http_failure(exc.response, provider)) from None
    except requests.Timeout:
        raise _ModelError(f'连接 {provider} 超时。') from None
    except requests.RequestException:
        raise _ModelError(f'连接 {provider} 失败，请检查网络。') from None
    except (KeyError, IndexError, TypeError, ValueError):
        raise _ModelError(f'{provider} 返回的答复格式无法读取。') from None
    return result if isinstance(result, dict) else {}


def model_connection_error(config):
    """Check a model credential with a fixed prompt and return a safe error if any."""
    messages = [{'role': 'user', 'content': 'Return only this JSON object: {"connected":true}.'}]
    try:
        result = _request_json(config, messages, 64)
    except _ModelError as exc:
        return str(exc)
    if result.get('connected') is not True:
        return f'{config.provider} 未返回有效的连接测试答复，请核对模型名。'


@lru_cache(maxsize=512)
def _explanation_order(payload, base_url, model_name):
    """The optional LLM can select sentence order. All claims are rendered locally."""
    messages = [
        {'role': 'system', 'content': '为新手安排两条已核验事实的阅读顺序。'
         '返回 JSON {"order":["rule","market"]} 或倒序。每个 ID 恰好出现一次。'},
        {'role': 'user', 'content': payload},
    ]
    try:
        config = ModelConfig(base_url, model_name, os.environ['OPENAI_API_KEY'].strip())
        order = _request_json(config, messages, 80, read_timeout=3).get('order')
    except _ModelError:
        return ('rule', 'market')
    if isinstance(order, list) and tuple(order) in (('rule', 'market'), ('market', 'rule')):
        return tuple(order)
    return ('rule', 'market')


def explain(facts, use_llm=False):
    parts = explanation_parts(facts)
    order = ('rule', 'market')
    config = model_from_env() if use_llm else None
    if config:
        order = _explanation_order(canonical(parts), config.base_url, config.model_name)
    return ' '.join(parts[name] for name in order)


@dataclass(frozen=True)
class Answer:
    text: str
    draft: bool = False
    target: float | None = None
    source: str = 'local'
    detail: str = ''


def _operation_reply(query, facts):
    match = TARGET.fullmatch(query)
    if match:
        number = Decimal(match.group(1))
        if number > 100:
            return Answer('请输入 0% 至 100% 的目标仓位。')
        target = float(number/100)
        return Answer(f"当前仓位 {facts['exposure']:.1%}。已生成目标 {target:.2%} 的调仓确认卡。",
                      True, target)
    if re.fullmatch(r'(?:(?:请|帮我|我想|我要|请帮我)\s*)?(?:保持(?:现有|原有|原)?份额|不调仓|维持现有份额)', query):
        return Answer('已生成保持原份额的确认卡。', True, None)
    if re.fullmatch(r'(?:(?:请|帮我|我想|我要)\s*)?(?:清仓|全部卖出)', query):
        return Answer('已生成目标仓位 0% 的确认卡。', True, 0.)
    if re.search(r'(?:加仓|减仓|卖出|买入)\s*\d+(?:\.\d+)?\s*%', query):
        return Answer('你说的是调整几个百分点，还是调整现有份额的百分比？'
                      '可以直接告诉我最终目标，例如「把仓位调到 35%」。')


def _fact_cards(facts):
    instrument = '合成行情中的 ETF' if facts.get('synthetic') else '510300'
    cards = {
        'decision': explain(facts),
        'market': f"截至 {facts['date']} 收盘，{instrument} 收报 {facts['close']:.3f} 元。"
                  f"近 1、5、20 个交易日的收盘价涨跌为 {facts['return_1']:+.2%}、"
                  f"{facts['return_5']:+.2%}、{facts['return_20']:+.2%}。"
                  f"50 日均线为 {facts['ma50']:.3f} 元。",
        'timing': f"本条建议的计划执行日为 {facts['execution_date']}，按该交易日开盘模拟处理。",
    }
    if 'equity' in facts:
        cards['account'] = (f"截至 {facts['date']} 收盘，账户净值 {facts['equity']:,.2f} 元，"
                            f"现金 {facts['cash']:,.2f} 元，持有 {facts['quantity']:,} 份，"
                            f"实际仓位 {facts['exposure']:.1%}。")
    return cards


def _local_reply(query, cards):
    if any(key in query for key in ('为什么','理由','依据','原因','为什么留')):
        return Answer(cards['decision'])
    if any(key in query for key in ('行情','走势','近5日','近五日','近20日','今天涨','今天跌')):
        return Answer(cards['market'])
    if 'account' in cards and any(key in query for key in ('我的账户','我的仓位','多少现金','持有多少','还有多少钱')):
        return Answer(cards['account'])
    if any(key in query for key in ('什么时候','何时','执行','开盘')):
        return Answer(cards['timing'])
    definitions = [value for name,value in TERMS.items() if name.casefold() in query.casefold()]
    if definitions:
        return Answer(' '.join(definitions[:3]))
    return Answer('我可以解释这条建议、账户术语或当前已揭示的行情。'
                  '要自选仓位，可以说「把仓位调到 35%」或「保持原份额」。')


def _model_reply(question, cards, history, config, *, facts_only):
    messages = [{'role': 'system', 'content':
                 '你是 510300 模拟交易台的中文问答助手。回答要直白，面向新手。'
                 '有关本轮建议、价格、账户或执行时间，只能选择已核验的事实卡，返回 JSON '
                 '{"kind":"facts","ids":["decision"]}，可选多个相关 ID。'
                 '解释通用术语时，可以返回 {"kind":"knowledge","text":"简短解释"}，'
                 '使用清楚的假设例子，不要写入该账户、该交易日或模型输出的具体事实。'
                 '不要预测未来价格、承诺收益、给个人投资结论或声称已提交操作。'
                 '无法核实时说明具体缺少什么。事实卡 JSON：' + canonical(cards)}]
    messages.extend({'role': role, 'content': content[:500]} for role, content in history[-6:]
                    if role in ('user', 'assistant'))
    messages.append({'role': 'user', 'content': question})
    result = _request_json(config, messages, 450)
    if result.get('kind') == 'facts':
        ids = result.get('ids')
        if isinstance(ids, list) and 1 <= len(ids) <= 4 and all(isinstance(key, str) and key in cards for key in ids):
            return ' '.join(cards[key] for key in dict.fromkeys(ids))
    if result.get('kind') == 'knowledge' and not facts_only:
        content = result.get('text')
        if isinstance(content, str) and 0 < len(content.strip()) <= 500:
            return content.strip()
    return None


def respond(message, facts, *, history=(), use_llm=False, model_config=None):
    query = message.strip().replace('％','%').rstrip('。！!').strip()
    if len(query) > 500:
        return Answer('请用 500 字以内描述你的问题或目标仓位。')
    operation = _operation_reply(query, facts)
    if operation is not None:
        return operation
    cards = _fact_cards(facts)
    local = _local_reply(query, cards)
    config = (model_config or model_from_env()) if use_llm else None
    if config:
        try:
            answer = _model_reply(query, cards, history, config, facts_only=local.text in cards.values())
        except _ModelError as exc:
            return Answer(local.text, source='fallback', detail=str(exc))
        if answer:
            return Answer(answer, source='model')
        return Answer(local.text, source='fallback', detail='模型回答未通过事实校验。')
    return local
