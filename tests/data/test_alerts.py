"""Email alerts, against a fake SMTP server: nothing is ever sent."""

import smtplib
from types import SimpleNamespace
from typing import ClassVar

import pytest

from primer.data import alerts


class FakeSMTP:
    """Same constructor as ``smtplib.SMTP``, which takes no ``context`` (only SMTP_SSL does)."""

    sent: ClassVar[list] = []
    instances: ClassVar[list] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.calls = host, port, []
        FakeSMTP.instances.append(self)

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, password):
        password.encode("ascii")   # like SMTP.auth: a non-ASCII password raises UnicodeEncodeError
        self.calls.append(("login", user))

    def send_message(self, msg):
        FakeSMTP.sent.append(msg)
        return {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.calls.append("closed")
        return False


class FakeSMTPSSL(FakeSMTP):
    def __init__(self, host, port, timeout=None, context=None):
        super().__init__(host, port, timeout)


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.sent, FakeSMTP.instances = [], []
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTPSSL)
    for var in ("PRIMER_ALERT_EMAILS", "PRIMER_SMTP_HOST", "PRIMER_SMTP_PORT", "PRIMER_SMTP_USER",
                "PRIMER_SMTP_PASSWORD", "PRIMER_ALERT_FROM"):
        monkeypatch.delenv(var, raising=False)
    return FakeSMTP


def configure(monkeypatch, **extra):
    env = {"PRIMER_ALERT_EMAILS": "a@example.com, b@example.com", "PRIMER_SMTP_USER": "bot@example.com",
           "PRIMER_SMTP_PASSWORD": "x", **extra}
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_off_until_recipients_are_set(smtp, caplog):
    assert not alerts.EmailConfig.from_env().enabled
    assert alerts.send_alert("hello", "body") is False
    assert smtp.instances == []                      # never connects
    assert "Email alerts are off" in caplog.text and "[primer] hello" in caplog.text


def test_sends_over_starttls_by_default(smtp, monkeypatch):
    configure(monkeypatch)
    assert alerts.send_alert("Data drift: 2 warnings", "details") is True
    [server] = smtp.instances
    assert (server.host, server.port) == ("smtp.gmail.com", 587)
    assert server.calls == ["starttls", ("login", "bot@example.com"), "closed"]
    [msg] = smtp.sent
    assert msg["Subject"] == "[primer] Data drift: 2 warnings"
    assert msg["To"] == "a@example.com, b@example.com" and msg["From"] == "bot@example.com"
    assert msg.get_content().strip() == "details"


def test_port_465_uses_ssl_without_starttls(smtp, monkeypatch):
    configure(monkeypatch, PRIMER_SMTP_PORT="465", PRIMER_SMTP_HOST="mail.example.com",
              PRIMER_ALERT_FROM="primer@example.com")
    assert alerts.send_alert("s", "b")
    [server] = smtp.instances
    assert (server.host, server.port) == ("mail.example.com", 465)
    assert "starttls" not in server.calls
    assert smtp.sent[0]["From"] == "primer@example.com"


def test_a_broken_mail_server_never_raises(smtp, monkeypatch, caplog):
    configure(monkeypatch)

    def refuse(self, user, password):
        raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    monkeypatch.setattr(FakeSMTP, "login", refuse)
    assert alerts.send_alert("s", "the alert text") is False
    assert "Could not send alert email" in caplog.text and "the alert text" in caplog.text


def test_an_unreachable_mail_server_never_raises(smtp, monkeypatch, caplog):
    configure(monkeypatch)

    def unreachable(self, host, port, timeout=None):
        raise ConnectionRefusedError(111, "Connection refused")   # an OSError, not an SMTPException

    monkeypatch.setattr(FakeSMTP, "__init__", unreachable)
    assert alerts.send_alert("s", "the alert text") is False
    assert "Could not send alert email" in caplog.text and "the alert text" in caplog.text


def test_a_failed_starttls_still_closes_the_connection(smtp, monkeypatch):
    configure(monkeypatch)

    def fail(self, context=None):
        raise smtplib.SMTPNotSupportedError("STARTTLS extension not supported by server.")

    monkeypatch.setattr(FakeSMTP, "starttls", fail)
    assert alerts.send_alert("s", "b") is False
    [server] = smtp.instances
    assert server.calls == ["closed"]


@pytest.mark.parametrize("env", [
    {"PRIMER_SMTP_PORT": "587  # starttls"},           # a trailing comment in .env: int() raises
    {"PRIMER_ALERT_FROM": "bot@example.com\nBcc: x"},  # a header with a newline: EmailMessage raises
    {"PRIMER_SMTP_PASSWORD": "pässword"},               # non-ASCII: login raises UnicodeEncodeError
])
def test_bad_settings_are_logged_not_raised(smtp, monkeypatch, caplog, env):
    configure(monkeypatch, **env)
    assert alerts.send_alert("s", "the alert text") is False
    assert "the alert text" in caplog.text
    assert smtp.sent == []


def test_partly_refused_recipients_are_logged(smtp, monkeypatch, caplog):
    configure(monkeypatch)
    monkeypatch.setattr(FakeSMTP, "send_message", lambda self, msg: {"b@example.com": (550, b"no such user")})
    assert alerts.send_alert("s", "b") is True
    assert "refused for b@example.com" in caplog.text


def test_task_failure_callback_emails_the_failure(smtp, monkeypatch):
    configure(monkeypatch)
    ti = SimpleNamespace(dag_id="primer_data_pipeline", task_id="quickdraw.validate", try_number=1,
                         log_url="http://localhost:8080/log")
    alerts.notify_task_failure({"ti": ti, "run_id": "manual__1", "exception": ValueError("3 rows without ids")})
    [msg] = smtp.sent
    assert msg["Subject"] == "[primer] Task failed: primer_data_pipeline.quickdraw.validate"
    body = msg.get_content()
    assert "manual__1" in body and "3 rows without ids" in body and "http://localhost:8080/log" in body


def test_a_multi_line_error_stays_readable_in_the_email(smtp, monkeypatch):
    configure(monkeypatch)
    ti = SimpleNamespace(dag_id="primer_data_pipeline", task_id="check_data", try_number=1)
    error = RuntimeError("2 schema errors in the data.\n- ERROR drawings [label]: a\n- ERROR words: b")
    alerts.notify_task_failure({"ti": ti, "run_id": "r", "exception": error})
    body = smtp.sent[0].get_content()
    assert "\n- ERROR drawings [label]: a\n- ERROR words: b" in body and "\\n" not in body
    assert "Error:  RuntimeError" in body
