"""本地分类接口。前端上传 user_id 和文本文件，返回待办、速记、打卡。

数据库只读：按 user_id 查询 user_plan_tasks 里 task_status 为 1 的未删除任务。
文案对上任务名，并且有完成打卡的意向时，额外返回「打卡」和「打卡成功」。

启动：
    python server.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from checkin import match_checkins
from classify import ClassifyError, classify_text, decode_text, load_dotenv, resolve_api_key
from db import DatabaseError, fetch_open_tasks, tasks_covering
from i18n import LABEL_ORDER, ApiError, display_labels, resolve_lang, result_msg, text

ROOT = Path(__file__).resolve().parent
MAX_FILE_BYTES = 1_000_000
SHANGHAI = timezone(timedelta(hours=8))

app = Flask(__name__)


def _json_error(code: str, status: int, lang: str):
    response = jsonify({"msg": text(lang, code)})
    response.status_code = status
    return response


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Lang"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


@app.get("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.route("/api/classify", methods=["OPTIONS"])
def classify_options():
    return ("", 204)


def parse_written_at(value: str) -> datetime:
    """解析文案书写时间，并换成北京时间后再和任务周期比较。"""
    raw = value.strip()
    if not raw:
        raise ApiError("written_at_required", 400)
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if " " in raw and "T" not in raw:
        raw = raw.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ApiError("written_at_format", 400) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SHANGHAI)
    return parsed.astimezone(SHANGHAI).replace(tzinfo=None)


def _request_text() -> tuple[str, bytes, str]:
    """从 JSON 或表单里取出 user_id、文案和文案书写时间。"""
    if request.is_json:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ApiError("json_object", 400)
        user_raw = str(body.get("user_id", "")).strip()
        content = body.get("text", "")
        written_at = body.get("written_at", "")
        if not isinstance(content, str) or not content.strip():
            raise ApiError("text_required", 400)
        if not isinstance(written_at, str):
            raise ApiError("written_at_required", 400)
        data = content.encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            raise ApiError("text_too_large", 400)
        return user_raw, data, written_at

    user_raw = (request.form.get("user_id") or "").strip()
    written_at = request.form.get("written_at") or ""
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ApiError("file_required", 400)
    data = upload.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ApiError("file_too_large", 400)
    return user_raw, data, written_at


@app.post("/api/classify")
def classify_upload():
    """接收 user_id 和文案，按 Lang 返回中文或英文结果。"""
    try:
        lang = resolve_lang(request.headers.get("Lang"))
    except ApiError:
        return _json_error("lang_invalid", 400, "zh-CN")
    try:
        user_raw, data, written_raw = _request_text()
        written_at = parse_written_at(written_raw)
    except ApiError as exc:
        return _json_error(exc.code, exc.status, lang)
    if not user_raw.isdigit():
        return _json_error("user_id_invalid", 400, lang)

    api_key = resolve_api_key()
    if not api_key:
        return _json_error("api_key_missing", 500, lang)
    try:
        content = decode_text(data, "上传的文件")
    except ClassifyError:
        return _json_error("text_empty", 400, lang)
    try:
        tasks = tasks_covering(fetch_open_tasks(int(user_raw)), written_at)
        base_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = os.environ.get("DASHSCOPE_MODEL", "qwen3.7-flash")
        labels, items = classify_text(
            content,
            api_key=api_key,
            base_url=base_url,
            model=model,
            output_lang=lang,
        )
        matched = match_checkins(
            content,
            tasks,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    except ClassifyError:
        return _json_error("model_failed", 502, lang)
    except DatabaseError:
        return _json_error("database_failed", 503, lang)

    if matched and "打卡" not in labels:
        labels.append("打卡")
    ordered = [label for label in LABEL_ORDER if label in labels]
    success = bool(matched)
    return jsonify(
        {
            "user_id": int(user_raw),
            "labels": display_labels(lang, ordered),
            "items": items,
            "checkin_success": success,
            "msg": result_msg(lang, checkin_success=success, labels=ordered),
            "matched_tasks": matched,
        }
    )


def main() -> None:
    load_dotenv(ROOT / ".env")
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
