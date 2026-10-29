"""把一段文案标成「待办」和「速记」，两个标签可以同时出现。多人对话不标成速记。

读取一个文本文件，调用千问文本模型 qwen3.7-flash。
先逐行打印命中的标签；有待办时，再每行打印一条事项。

用法：
    python classify.py note.txt

密钥从环境变量 DASHSCOPE_API_KEY 读取，也可写在同目录 .env 里。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "qwen3.7-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LABELS = ("待办", "速记")
_ITEM_PREFIX = re.compile(r"^(?:[-*•·]+|\d+\s*[.、．)）]\s*)")

SYSTEM_PROMPT = """你是文案分类与待办抽取器。阅读用户给出的一整段文字。同一段可以同时是「待办」和「速记」。不要输出解释。

判定规则：
1. 待办：文字里有即将要做的事，包括计划、安排、打算、提醒，以及要去完成的具体事项。能抽出至少一条，就带上待办。多个说话人里的待办照常识别。
2. 速记只来自非多人对话、且识别不出待办的文字。心情、感受、评价、见闻、已经发生或正在发生的描述、没有后续行动的闲聊，都是速记。例如「我心情很好」。
3. 多人对话不算速记。对话里的问句、寒暄、接话、叮嘱、已经发生的事，都不要因此输出速记。
4. 整段都是多人对话，并且抽不出待办时，不要输出待办，也不要输出速记，只输出一行：无。
5. 整段都抽不出待办、并且不是多人对话时，只输出速记。
6. 整段全是即将要做的事、没有非对话的其余文字时，只输出待办。
7. 一段里既有即将要做的事，又有非多人对话的速记内容时，两个标签都输出。例如「我心情很好。我下午要吃饭，明天要睡觉。」同时是待办和速记。多人对话不会单独带来速记。
8. 已经做完的事、单纯的愿望或反事实，不算待办。这些文字只有在不是多人对话时才算速记。例如「我吃过饭了」「我要是早知道就好了」。
9. 问句、对别人的叮嘱，如果推不出某人即将亲自去做的具体事，不要写成待办事项。不要猜测原文没说的行动。它们出现在多人对话里时不算速记。

输出格式：
- 先逐行输出标签，标签只能是「待办」或「速记」。两个都有时先写待办，再写速记。
- 有待办时，标签写完后每行一条事项，不要编号、项目符号、引号或解释。
- 只有速记时，只输出一行：速记

事项怎么写：
1. 一段里有多件即将要做的事，拆成多条。
2. 没有点名说话人时，不写「我」。有时间就写成「时间 + 要 + 做什么」，例如「下午要吃饭」「明天要睡觉」。没有时间就只写做法，例如「买牛奶」。
3. 多人对话按语义写成「谁 + 什么时候 + 要 + 做什么」，谁用原文里的称呼。例如「乙下午要交报告」。没有时间就写成「谁要做什么」，例如「乙要交报告」。
4. 什么时候只有原文能确定时才写，确定不了就省略，不要编时间。
5. 用短句保住原意，把「把报告交了」收成「交报告」这类做法，不要扩写。

示例：
输入：我心情很好。我下午要吃饭，明天要睡觉。
输出：
待办
速记
下午要吃饭
明天要睡觉

输入：我下午要吃饭，明天要睡觉。
输出：
待办
下午要吃饭
明天要睡觉

输入：记得买牛奶。
输出：
待办
买牛奶

输入：
甲：你下午有安排吗？
乙：有，我要去把报告交了。
丙：那晚饭别等我。
输出：
待办
乙下午要交报告

输入：
甲：报告呢？
乙：我去交。
输出：
待办
乙要交报告

输入：
甲：你下午干什么？
乙：我要去医院拿药。
输出：
待办
乙下午要去医院拿药

输入：
甲：今天天气真好。
乙：是啊，出去走走吧。
输出：
无

输入：我心情很好。
甲：你下午干什么？
乙：我要去医院拿药。
输出：
待办
速记
乙下午要去医院拿药

输入：我心情很好
输出：
速记

输入：今天阳光很好，坐在窗边很舒服。
输出：
速记
"""


class ClassifyError(Exception):
    """调用模型或解析标签失败。"""


def load_dotenv(path: Path) -> None:
    """把 .env 里尚未设置的变量写入环境，不覆盖已有值。"""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def read_text(path: Path) -> str:
    data = path.read_bytes()
    if not data.strip():
        raise ClassifyError(f"文件是空的：{path}")
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", errors="replace")
    text = text.strip()
    if not text:
        raise ClassifyError(f"文件里没有可分类的文字：{path}")
    return text


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def split_label_line(line: str) -> tuple[str | None, str | None]:
    """从一行里取出标签；标签后面如果跟着事项，一并返回。"""
    text = line.strip().strip("。．.\"'“”‘’` \t")
    if not text:
        return None, None
    for label in LABELS:
        if text == label:
            return label, None
        if text.startswith(label):
            rest = text[len(label):].strip(" ：:\t")
            return label, rest or None
    return None, None


def clean_item(line: str) -> str:
    text = _ITEM_PREFIX.sub("", line.strip())
    return text.strip().strip("。．.\"'“”‘’` \t")


_LABEL_SPLIT = re.compile(r"[、,，/|+\s和与及]+")


def labels_only(line: str) -> list[str]:
    """一行里只有标签时返回这些标签，否则返回空列表。"""
    text = line.strip().strip("。．.\"'“”‘’` \t")
    if not text:
        return []
    found: list[str] = []
    remaining = text
    while remaining:
        matched = next((label for label in LABELS if remaining.startswith(label)), None)
        if matched is None:
            return []
        if matched not in found:
            found.append(matched)
        remaining = _LABEL_SPLIT.sub("", remaining[len(matched):], count=1)
        remaining = remaining.lstrip("、,，/|+和与及 \t")
    return found


def parse_result(raw: str) -> tuple[list[str], list[str]]:
    """解析模型回复，返回标签列表和待办事项。没有待办时事项列表为空。"""
    text = raw.strip()
    if text.startswith("```"):
        kept = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(kept)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ClassifyError("模型没有返回内容")

    labels: list[str] = []
    items: list[str] = []
    for line in lines:
        pure = labels_only(line)
        if pure:
            for label in pure:
                if label not in labels:
                    labels.append(label)
            continue
        label, inline_item = split_label_line(line)
        if label and inline_item:
            if label not in labels:
                labels.append(label)
            item = clean_item(inline_item)
            if item and item not in LABELS and item not in items:
                items.append(item)
            continue
        item = clean_item(line)
        if item == "无" or not item or item in LABELS or item in items:
            continue
        items.append(item)

    if not labels:
        if not items:
            return [], []
        shown = raw.strip() or "（空回复）"
        raise ClassifyError(f"模型没有返回有效标签：{shown}")

    ordered = [label for label in LABELS if label in labels]
    if "待办" in ordered and not items:
        raise ClassifyError(f"识别为待办，但没有抽出事项：{raw.strip()}")
    if "待办" not in ordered:
        items = []
    return ordered, items


def classify_text(
    text: str,
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = MODEL,
    timeout: float = 120,
) -> tuple[list[str], list[str]]:
    """调用 qwen3.7-flash，返回标签列表和待办事项。没有待办时事项列表为空。"""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": 512,
        "enable_thinking": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise ClassifyError(f"模型接口返回 {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise ClassifyError(f"无法连接模型接口：{exc.reason}") from exc

    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ClassifyError(f"模型返回格式异常：{body}") from exc
    return parse_result(message_text(message))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用千问 qwen3.7-flash 判断一段文案是「待办」「速记」或两者都是，并抽出待办事项。",
    )
    parser.add_argument("file", type=Path, help="待分类的文本文件，文件里是一整段文字")
    parser.add_argument(
        "--model",
        default=os.environ.get("DASHSCOPE_MODEL", MODEL),
        help=f"模型名，默认 {MODEL}",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI 兼容接口地址",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="百炼 API Key，默认读环境变量 DASHSCOPE_API_KEY",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    load_dotenv(Path(__file__).resolve().parent / ".env")
    args = build_parser().parse_args(argv)
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY", "sk-ws-H.YDPYELE.TVGQ.MEQCIGE_55g6r0KNzs0jwhHB1zBBknGUIA6xuX1Ht6_riJGZAiB8HgvY_aF5hEOpappQFn8MEmsn98wuGOtsg7wyXJRcNA")
    if not api_key:
        print("缺少 DASHSCOPE_API_KEY。请在环境变量或 .env 中填写百炼 API Key。", file=sys.stderr)
        return 1
    if not args.file.is_file():
        print(f"找不到文件：{args.file}", file=sys.stderr)
        return 1
    try:
        text = read_text(args.file)
        labels, items = classify_text(
            text,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
        )
    except ClassifyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for label in labels:
        print(label)
    for item in items:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
