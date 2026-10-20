"""插件式提醒。配置全走环境变量；没配就跳过。不发真实持仓细节。"""

from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from email.message import EmailMessage
from typing import Protocol


class Alerter(Protocol):
    def send(self, title: str, body: str) -> str: ...


def webhook_send(title: str, body: str) -> str:
    url = os.environ.get("ZHIKU_WEBHOOK_URL")
    if not url:
        return "跳过 webhook：未配置 ZHIKU_WEBHOOK_URL"
    payload = json.dumps({"title": title, "text": body, "content": f"{title}\n{body}"}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "zhiku-alert"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
    return "webhook 已发送"


def telegram_send(title: str, body: str) -> str:
    token = os.environ.get("ZHIKU_TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("ZHIKU_TELEGRAM_CHAT_ID")
    if not token or not chat:
        return "跳过 Telegram：未配置 ZHIKU_TELEGRAM_BOT_TOKEN / ZHIKU_TELEGRAM_CHAT_ID"
    text = urllib.parse.quote(f"{title}\n{body}"[:3500])
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat}&text={text}"
    urllib.request.urlopen(url, timeout=10).read()
    return "telegram 已发送"


def email_send(title: str, body: str) -> str:
    host = os.environ.get("ZHIKU_SMTP_HOST")
    to = os.environ.get("ZHIKU_ALERT_EMAIL")
    if not host or not to:
        return "跳过邮件：未配置 ZHIKU_SMTP_HOST / ZHIKU_ALERT_EMAIL"
    user = os.environ.get("ZHIKU_SMTP_USER", "")
    password = os.environ.get("ZHIKU_SMTP_PASSWORD", "")
    port = int(os.environ.get("ZHIKU_SMTP_PORT", "587"))
    msg = EmailMessage()
    msg["Subject"] = title
    msg["From"] = user or "zhiku@localhost"
    msg["To"] = to
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=10) as smtp:
        smtp.starttls()
        if user:
            smtp.login(user, password)
        smtp.send_message(msg)
    return "邮件已发送"


def dispatch(title: str, body: str) -> list[str]:
    notes: list[str] = []
    for fn in (webhook_send, telegram_send, email_send):
        try:
            notes.append(fn(title, body))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{fn.__name__} 失败: {exc}")
    return notes


# urllib.parse used in telegram_send
import urllib.parse  # noqa: E402
