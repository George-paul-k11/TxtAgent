"""根据文案生成结构化笔记，供 /api/classify 返回。"""

from __future__ import annotations

import json
import re
from typing import Any

from classify import ClassifyError, chat_completion

NOTE_SECTION_KINDS = (
    "title",
    "introduction",
    "summary",
    "key_points",
    "decisions",
    "chapters",
    "todos",
)

_TIMESTAMP = re.compile(
    r"\[\s*(?P<start>\d{2}:\d{2}\.\d+)\s*-\s*(?P<end>\d{2}:\d{2}\.\d+)\s*\]"
    r"(?P<speaker>[^:：\[]+?)[:：](?P<body>[^\[]*)",
)

_SYSTEM_ZH = """你是会议/对话笔记助手。根据用户给出的原文，生成结构化笔记。只输出一个 JSON 对象，不要 markdown、不要解释。

JSON 字段（键名必须完全一致）：
- title：结合全文总结的标题，一句话。
- introduction：引言，2～4 句概括背景与主题。
- summary：摘要，一段连贯文字。
- key_points：要点，字符串数组，每条一个独立要点，不要编号前缀。
- decisions：决策或结论，字符串数组；没有则 []。
- chapters：按原文时间轴归纳的章节（可合并相邻、同一主题的细粒度时间戳）。仅当原文含 [00:00.000 -00:00.000] 这类时间戳时使用；每项为对象：start_time、end_time、title、text。start_time/end_time 取该段起止时间（与原文一致即可，后端会格式化为 MM:SS）。title 为该段主题短标题（如「蓝牙配对与重连异常排查」）。text 为该时段讨论内容的润色摘要，保留原意。无时间戳则 []。
- todos：从原文抽出的待办，字符串数组；没有则 []。

chapters 示例原文：
[00:00.340 -00:01.190]说话人1:从前有座山，[00:01.190-00:02.220]说话人2：山上有座庙
应对应两条 chapter，时间戳与说话人取自原文。"""

_SYSTEM_EN = """You are a note-taking assistant. From the user's transcript, produce structured notes. Output only one JSON object. No markdown, no extra text.

Keys (exact names):
- title: one-line title summarizing the content.
- introduction: 2–4 sentences of context.
- summary: one paragraph summary.
- key_points: array of bullet strings, no numbering prefixes.
- decisions: array of decision/conclusion strings, or [] if none.
- chapters: only when the source has [HH:MM.mmm -HH:MM.mmm] timestamps; merge adjacent segments on the same topic when appropriate. Each item: start_time, end_time, title (short topic), text (summary). Otherwise [].
- todos: action items from the text, or [] if none."""

_EN_GUIDE = "Write all string values in English."
_ZH_GUIDE = "所有字符串值用中文书写。"


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def extract_chapter_hints(content: str) -> list[dict[str, str]]:
    """从带时间戳的原文里抽出章节骨架，供模型对齐。"""
    hints: list[dict[str, str]] = []
    for match in _TIMESTAMP.finditer(content):
        body = match.group("body").strip().strip("，,。．.")
        hints.append(
            {
                "start_time": match.group("start"),
                "end_time": match.group("end"),
                "speaker": match.group("speaker").strip(),
                "text": body,
            }
        )
    return hints


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text and text not in out:
                out.append(text)
    return out


def _display_time(value: str) -> str:
    """00:00.340 -> 00:00，供章节行展示。"""
    raw = value.strip()
    if not raw:
        return ""
    head = raw.split(".", 1)[0]
    if ":" not in head:
        return head
    minutes, seconds = head.split(":", 1)
    try:
        return f"{int(minutes):02d}:{int(seconds):02d}"
    except ValueError:
        return head


def _normalize_chapters(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    chapters: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        start = str(item.get("start_time") or item.get("start") or "").strip()
        end = str(item.get("end_time") or item.get("end") or "").strip()
        title = str(item.get("title") or item.get("topic") or "").strip()
        text = str(item.get("text") or item.get("content") or "").strip()
        if not text and not start:
            continue
        chapters.append(
            {
                "start_time": start,
                "end_time": end,
                "title": title,
                "text": text,
            }
        )
    return chapters


def format_chapter_line(chapter: dict[str, str]) -> str:
    """组装为前端展示行：00:00~05:09主题标题：摘要内容。"""
    start = _display_time(chapter.get("start_time", ""))
    end = _display_time(chapter.get("end_time", ""))
    title = (chapter.get("title") or "").strip()
    body = (chapter.get("text") or "").strip()
    if start and end:
        prefix = f"{start}~{end}"
    elif start:
        prefix = start
    else:
        prefix = ""
    if title and body:
        return f"{prefix}{title}：{body}"
    if title:
        return f"{prefix}{title}"
    if prefix and body:
        return f"{prefix}：{body}"
    return body or prefix


def parse_note_json(raw: str) -> dict[str, Any]:
    text = _strip_json_fence(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClassifyError(f"笔记 JSON 解析失败：{raw.strip()[:200]}") from exc
    if not isinstance(data, dict):
        raise ClassifyError("笔记返回不是 JSON 对象")

    title = str(data.get("title") or "").strip()
    introduction = str(data.get("introduction") or "").strip()
    summary = str(data.get("summary") or "").strip()
    if not title:
        raise ClassifyError("笔记缺少 title")

    return {
        "title": title,
        "introduction": introduction,
        "summary": summary,
        "key_points": _as_str_list(data.get("key_points")),
        "decisions": _as_str_list(data.get("decisions")),
        "chapters": _normalize_chapters(data.get("chapters")),
        "todos": _as_str_list(data.get("todos")),
    }


def note_to_blocks(note: dict[str, Any]) -> list[dict[str, Any]]:
    """带 kind 枚举的区块列表，便于前端按类型渲染。"""
    chapter_lines = [format_chapter_line(ch) for ch in note["chapters"]]
    blocks: list[dict[str, Any]] = [
        {"kind": "title", "content": note["title"]},
        {"kind": "introduction", "content": note["introduction"]},
        {"kind": "summary", "content": note["summary"]},
        {"kind": "key_points", "content": note["key_points"]},
        {"kind": "decisions", "content": note["decisions"]},
        {"kind": "chapters", "content": chapter_lines},
        {"kind": "todos", "content": note["todos"]},
    ]
    return blocks


def generate_note(
    text: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float = 120,
    output_lang: str = "zh-CN",
    fallback_todos: list[str] | None = None,
) -> dict[str, Any]:
    """调用模型生成笔记；待办为空时用分类结果 fallback_todos 填充。"""
    system = _SYSTEM_EN if output_lang == "en-US" else _SYSTEM_ZH
    guide = _EN_GUIDE if output_lang == "en-US" else _ZH_GUIDE
    hints = extract_chapter_hints(text)
    user_parts = [guide, "", "文案：", text]
    if hints:
        user_parts.extend(["", "时间轴片段（请据此填写 chapters，并润色 text）：", json.dumps(hints, ensure_ascii=False)])
    raw = chat_completion(
        system,
        "\n".join(user_parts),
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_tokens=2048,
    )
    note = parse_note_json(raw)
    if hints and not note["chapters"]:
        note["chapters"] = [
            {
                "start_time": h["start_time"],
                "end_time": h["end_time"],
                "title": "",
                "text": h["text"],
            }
            for h in hints
        ]
    if not note["todos"] and fallback_todos:
        note["todos"] = list(fallback_todos)
    return note
