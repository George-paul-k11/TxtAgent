"""判断文案是否在为用户的进行中任务打卡。"""

from __future__ import annotations

import re

from classify import DEFAULT_BASE_URL, MODEL, chat_completion

CHECKIN_PROMPT = """你负责判断文案是不是在给给定任务打卡。不要输出解释。

一条任务必须同时满足两点才算打卡：
1. 文案说的是这件任务。可以是任务全名、简称，或明显指同一件事。不要因为都含有同一个词，就把所有相近任务都算上。
2. 文案有完成打卡的意向，例如打卡、完成了、做完了、搞定了、已经做了。
只是计划、提醒、将要去做，不算打卡。例如「我下午要去跑步」不是打卡。
多人对话不影响判断。

输出：
- 没有符合的任务时，只输出一行：无
- 有则每行只输出一个任务名称，必须与给定名称完全一致，不要编号和解释。

示例：
任务：跑步训练、每周跑步、每天阅读
文案：今天跑步训练打卡了
输出：
跑步训练

任务：跑步训练、每周跑步、每天阅读
文案：我下午要去跑步
输出：
无

任务：跑步训练、每周跑步、每天阅读
文案：每周跑步完成了，今天也把每天阅读打卡了
输出：
每周跑步
每天阅读
"""

_ITEM_PREFIX = re.compile(r"^(?:[-*•·]+|\d+\s*[.、．)）]\s*)")
_IGNORE = {"无", "打卡", "打卡成功"}


def parse_matched_tasks(raw: str, tasks: list[dict]) -> list[dict]:
    """只保留模型点名、且确实存在于任务列表中的任务。"""
    text = raw.strip()
    if text.startswith("```"):
        kept = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(kept)

    by_name: dict[str, list[dict]] = {}
    for task in tasks:
        by_name.setdefault(task["task_name"], []).append(task)

    matched: list[dict] = []
    seen: set[int] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("。．.\"'“”‘’` \t")
        line = _ITEM_PREFIX.sub("", line).strip()
        if not line or line in _IGNORE:
            continue
        hits = by_name.get(line)
        if hits is None:
            contained = [name for name in by_name if name and name in line]
            if len(contained) != 1:
                continue
            hits = by_name[contained[0]]
        for task in hits:
            if task["id"] in seen:
                continue
            seen.add(task["id"])
            matched.append(task)
    return matched


def match_checkins(
    text: str,
    tasks: list[dict],
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = MODEL,
    timeout: float = 120,
) -> list[dict]:
    """返回文案有完成打卡意向、且对得上的任务。没有进行中任务时不调用模型。"""
    if not tasks:
        return []
    names = "\n".join(f"- {task['task_name']}" for task in tasks)
    user = f"任务：\n{names}\n\n文案：\n{text}"
    raw = chat_completion(
        CHECKIN_PROMPT,
        user,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_tokens=256,
    )
    return parse_matched_tasks(raw, tasks)
