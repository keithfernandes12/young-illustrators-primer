"""Email alerts for pipeline failures and data anomalies.

Plain ``smtplib``, configured from environment variables (put them in `.env`, which the
Airflow containers load):

    PRIMER_ALERT_EMAILS     comma-separated recipients; alerts are off while this is empty
    PRIMER_SMTP_HOST        default smtp.gmail.com
    PRIMER_SMTP_PORT        default 587 (STARTTLS); 465 uses SSL
    PRIMER_SMTP_USER        the sending account
    PRIMER_SMTP_PASSWORD    for Gmail, an app password (Google account → Security → App passwords)
    PRIMER_ALERT_FROM       default PRIMER_SMTP_USER

When alerts are off, or sending fails, the alert is logged instead. An alert never raises:
a broken mail server must not fail a pipeline task or hide the error being reported.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any

log = logging.getLogger(__name__)

SUBJECT_PREFIX = "[primer]"


@dataclass(frozen=True)
class EmailConfig:
    recipients: tuple[str, ...]
    host: str = "smtp.gmail.com"
    port: int = 587
    user: str = ""
    password: str = ""
    sender: str = ""

    @classmethod
    def from_env(cls) -> EmailConfig:
        env = os.environ
        user = env.get("PRIMER_SMTP_USER", "").strip()
        return cls(
            recipients=tuple(r.strip() for r in env.get("PRIMER_ALERT_EMAILS", "").split(",") if r.strip()),
            host=env.get("PRIMER_SMTP_HOST", "").strip() or cls.host,
            port=int(env.get("PRIMER_SMTP_PORT", "").strip() or cls.port),
            user=user,
            password=env.get("PRIMER_SMTP_PASSWORD", ""),
            sender=env.get("PRIMER_ALERT_FROM", "").strip() or user,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.recipients and self.sender)


def send_alert(subject: str, body: str, config: EmailConfig | None = None) -> bool:
    """Email ``body`` to the alert recipients. Returns True if it was sent."""
    config = config or EmailConfig.from_env()
    subject = f"{SUBJECT_PREFIX} {subject}"
    if not config.enabled:
        log.warning("Email alerts are off (set PRIMER_ALERT_EMAILS and PRIMER_SMTP_USER). Alert: %s\n%s",
                    subject, body)
        return False

    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, config.sender, ", ".join(config.recipients)
    msg.set_content(body)
    try:
        if config.port == 465:
            server: smtplib.SMTP = smtplib.SMTP_SSL(config.host, config.port, timeout=30,
                                                    context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(config.host, config.port, timeout=30)
            server.starttls(context=ssl.create_default_context())
        with server:
            if config.user:
                server.login(config.user, config.password)
            server.send_message(msg)
    except (OSError, smtplib.SMTPException) as e:
        log.error("Could not send alert email via %s:%s (%r). Alert: %s\n%s",
                  config.host, config.port, e, subject, body)
        return False
    log.info("Alert emailed to %s: %s", ", ".join(config.recipients), subject)
    return True


def task_failure_body(context: dict[str, Any]) -> tuple[str, str]:
    """Subject and body for a failed Airflow task, from its callback context."""
    ti = context["ti"]
    task = f"{ti.dag_id}.{ti.task_id}"
    lines = [
        f"Task:   {task}",
        f"Run:    {context.get('run_id')}",
        f"Try:    {getattr(ti, 'try_number', '?')}",
        f"Error:  {context.get('exception')!r}",
    ]
    if url := getattr(ti, "log_url", None):
        lines.append(f"Logs:   {url}")
    return f"Task failed: {task}", "\n".join(lines)


def notify_task_failure(context: dict[str, Any]) -> None:
    """Airflow ``on_failure_callback``: log the failure and email it."""
    subject, body = task_failure_body(context)
    log.error("%s\n%s", subject, body)
    send_alert(subject, body)
