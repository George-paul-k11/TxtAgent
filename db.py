"""只读访问 app-dev.user_plan_tasks。算法端不写数据库。"""

from __future__ import annotations

import os
from datetime import datetime

import pymysql
import pymysql.cursors


class DatabaseError(Exception):
    """读取任务失败。"""


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise DatabaseError(f"缺少数据库配置 {name}")
    return value


def fetch_open_tasks(user_id: int) -> list[dict]:
    """读取该用户 task_status 为 1、且未删除的任务。只执行查询。"""
    try:
        connection = pymysql.connect(
            host=_required("MYSQL_HOST"),
            port=int(os.environ.get("MYSQL_PORT", "3306")),
            user=_required("MYSQL_USER"),
            password=_required("MYSQL_PASSWORD"),
            database=_required("MYSQL_DATABASE"),
            charset="utf8mb4",
            connect_timeout=10,
            read_timeout=15,
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
        )
    except pymysql.MySQLError as exc:
        raise DatabaseError(f"无法连接数据库：{exc}") from exc

    sql = """
        SELECT id, task_name, period_start, period_end
        FROM user_plan_tasks
        WHERE user_id = %s AND task_status = 1 AND is_delete = 0
        ORDER BY id
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION TRANSACTION READ ONLY")
            cursor.execute(sql, (user_id,))
            rows = cursor.fetchall()
    except pymysql.MySQLError as exc:
        raise DatabaseError(f"读取任务失败：{exc}") from exc
    finally:
        connection.close()

    tasks = []
    for row in rows:
        name = str(row.get("task_name") or "").strip()
        if not name:
            continue
        tasks.append(
            {
                "id": int(row["id"]),
                "task_name": name,
                "period_start": row["period_start"],
                "period_end": row["period_end"],
            }
        )
    return tasks


def tasks_covering(tasks: list[dict], written_at: datetime) -> list[dict]:
    """只保留文案书写时间落在周期内的任务。两端都包含。"""
    covered = []
    for task in tasks:
        start = task["period_start"]
        end = task["period_end"]
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            continue
        if start <= written_at <= end:
            covered.append({"id": task["id"], "task_name": task["task_name"]})
    return covered
