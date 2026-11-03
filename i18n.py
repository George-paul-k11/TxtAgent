"""按请求头 Lang 选择返回文案。"""

from __future__ import annotations

ZH = "zh-CN"
EN = "en-US"

LABEL_ORDER = ("待办", "速记", "打卡")

_LABELS = {
    ZH: {"待办": "待办", "速记": "速记", "打卡": "打卡"},
    EN: {"待办": "To-do", "速记": "Note", "打卡": "Check-in"},
}

_TEXT = {
    ZH: {
        "checkin_ok": "打卡成功",
        "classified": "识别完成",
        "none": "未识别到待办、速记或打卡",
        "json_object": "请求体需要是 JSON 对象",
        "text_required": "text 需要是非空字符串",
        "written_at_required": "written_at 需要是带时区的文案书写时间，例如 2026-10-29T16:47:00+08:00",
        "written_at_format": "written_at 格式无法识别，请使用带时区的时间，例如 2026-10-29T16:47:00+08:00",
        "file_required": "请上传文本文件，字段名为 file",
        "file_too_large": "文本文件超过 1MB",
        "text_too_large": "文本超过 1MB",
        "text_empty": "文案是空的",
        "user_id_invalid": "user_id 需要是数字",
        "api_key_missing": "缺少模型密钥",
        "model_failed": "模型调用失败",
        "database_failed": "数据库读取失败",
        "lang_invalid": "Lang 只支持 zh-CN 或 en-US",
    },
    EN: {
        "checkin_ok": "Check-in successful",
        "classified": "Classification completed",
        "none": "No to-do, note, or check-in identified",
        "json_object": "Request body must be a JSON object",
        "text_required": "text must be a non-empty string",
        "written_at_required": "written_at is required and must include a timezone, for example 2026-10-29T08:47:00Z",
        "written_at_format": "written_at could not be parsed. Use a timezone, for example 2026-10-29T08:47:00Z",
        "file_required": "Upload a text file in the file field",
        "file_too_large": "The text file exceeds 1MB",
        "text_too_large": "The text exceeds 1MB",
        "text_empty": "The text is empty",
        "user_id_invalid": "user_id must be a number",
        "api_key_missing": "The model API key is missing",
        "model_failed": "The model request failed",
        "database_failed": "Failed to read the database",
        "lang_invalid": "Lang must be zh-CN or en-US",
    },
}


class ApiError(Exception):
    """可按 Lang 翻译的接口错误。"""

    def __init__(self, code: str, status: int):
        super().__init__(code)
        self.code = code
        self.status = status


def resolve_lang(value: str | None) -> str:
    """空值默认中文。无法识别的取值交给调用方返回 400。"""
    text = (value or "").strip()
    if not text:
        return ZH
    folded = text.replace("_", "-").lower()
    if folded in {"zh-cn", "zh"}:
        return ZH
    if folded in {"en-us", "en"}:
        return EN
    raise ApiError("lang_invalid", 400)


def text(lang: str, code: str) -> str:
    table = _TEXT.get(lang) or _TEXT[ZH]
    return table[code]


def display_labels(lang: str, labels: list[str]) -> list[str]:
    mapping = _LABELS.get(lang) or _LABELS[ZH]
    return [mapping[label] for label in labels if label in mapping]


def result_msg(lang: str, *, checkin_success: bool, labels: list[str]) -> str:
    if checkin_success:
        return text(lang, "checkin_ok")
    if labels:
        return text(lang, "classified")
    return text(lang, "none")
