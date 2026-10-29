"""本地分类接口。前端上传 user_id 和文本文件，返回待办、速记、打卡。

数据库只读：按 user_id 查询 user_plan_tasks 里 task_status 为 1 的未删除任务。
文案对上任务名，并且有完成打卡的意向时，额外返回「打卡」和「打卡成功」。

启动：
    python server.py
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request, send_from_directory

from checkin import match_checkins
from classify import ClassifyError, classify_text, decode_text, load_dotenv, resolve_api_key
from db import DatabaseError, fetch_open_tasks, tasks_covering

ROOT = Path(__file__).resolve().parent
MAX_FILE_BYTES = 1_000_000
LABEL_ORDER = ("待办", "速记", "打卡")

app = Flask(__name__)


def _json_error(message: str, status: int):
    response = jsonify({"error": message})
    response.status_code = status
    return response


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


@app.get("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.route("/api/classify", methods=["OPTIONS"])
def classify_options():
    return ("", 204)


def parse_written_at(value: str) -> datetime:
    """解析文案书写时间。带时区时按北京时间比较。"""
    text = value.strip()
    if not text:
        raise ClassifyError("written_at 需要是文案书写时间，格式如 2026-10-29 16:47:00")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ClassifyError("written_at 格式无法识别，请使用 2026-10-29 16:47:00")
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
    return parsed


def _request_text() -> tuple[str, bytes, str]:
    """从 JSON 或表单里取出 user_id、文案和文案书写时间。"""
    if request.is_json:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ClassifyError("请求体需要是 JSON 对象")
        user_raw = str(body.get("user_id", "")).strip()
        text = body.get("text", "")
        written_at = body.get("written_at", "")
        if not isinstance(text, str) or not text.strip():
            raise ClassifyError("text 需要是非空字符串")
        if not isinstance(written_at, str):
            raise ClassifyError("written_at 需要是文案书写时间，格式如 2026-10-29 16:47:00")
        data = text.encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            raise ClassifyError("文本超过 1MB")
        return user_raw, data, written_at

    user_raw = (request.form.get("user_id") or "").strip()
    written_at = request.form.get("written_at") or ""
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ClassifyError("请上传文本文件，字段名为 file")
    data = upload.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ClassifyError("文本文件超过 1MB")
    return user_raw, data, written_at


@app.post("/api/classify")
def classify_upload():
    """接收 user_id 和文案，返回三个标签、待办事项和打卡结果。"""
    try:
        user_raw, data, written_raw = _request_text()
        written_at = parse_written_at(written_raw)
    except ClassifyError as exc:
        return _json_error(str(exc), 400)
    if not user_raw.isdigit():
        return _json_error("user_id 需要是数字", 400)

    api_key = resolve_api_key()
    if not api_key:
        return _json_error("缺少 DASHSCOPE_API_KEY", 500)
    try:
        text = decode_text(data, "上传的文件")
        tasks = tasks_covering(fetch_open_tasks(int(user_raw)), written_at)
        labels, items = classify_text(
            text,
            api_key=api_key,
            base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=os.environ.get("DASHSCOPE_MODEL", "qwen3.7-flash"),
        )
        matched = match_checkins(
            text,
            tasks,
            api_key=api_key,
            base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=os.environ.get("DASHSCOPE_MODEL", "qwen3.7-flash"),
        )
    except ClassifyError as exc:
        return _json_error(str(exc), 502)
    except DatabaseError as exc:
        return _json_error(str(exc), 503)

    if matched and "打卡" not in labels:
        labels.append("打卡")
    ordered = [label for label in LABEL_ORDER if label in labels]
    success = bool(matched)
    return jsonify(
        {
            "user_id": int(user_raw),
            "labels": ordered,
            "items": items,
            "checkin_success": success,
            "message": "打卡成功" if success else "",
            "matched_tasks": matched,
        }
    )


def main() -> None:
    load_dotenv(ROOT / ".env")
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
