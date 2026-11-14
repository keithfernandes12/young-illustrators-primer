"""privacy_lint.py must flag private data that is sent somewhere, and ignore words in messages.

The kit's original rule matched the raw line, so `log.warning("Email alerts are off ...")` in
src/primer/data/alerts.py counted as a finding and verify.py printed FAIL. Every loop prompt gates
on VERIFY_RESULT: PASS, so that one regex stopped the loops working. The first test below is that
exact line.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".claude" / "skills" / "privacy-guard" / "scripts" / "privacy_lint.py"
spec = importlib.util.spec_from_file_location("privacy_lint", SCRIPT)
assert spec and spec.loader
privacy_lint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(privacy_lint)


def findings_for(source: str) -> list[str]:
    return privacy_lint.python_log_findings(Path("sample.py"), source, source.splitlines())


@pytest.mark.parametrize(
    "source",
    [
        # The real alerts.py lines, which used to produce five findings between them.
        'log.exception("Email alert settings are invalid. Alert: %s\\n%s", subject, body)',
        'log.warning("Email alerts are off (set PRIMER_ALERT_EMAILS and PRIMER_SMTP_USER).")',
        'log.error("Alert email refused for %s: %s", ", ".join(refused), subject)',
        'log.info("Alert emailed to %s: %s", ", ".join(config.recipients), subject)',
        # A message may describe the field without carrying it.
        'log.info("no child_name on this record, skipping")',
        'print("usage: primer-data --email-report")',
    ],
)
def test_privacy_words_in_a_message_are_not_findings(source):
    assert findings_for(source) == []


@pytest.mark.parametrize(
    "source",
    [
        'log.info("page for %s", child_name)',
        'log.debug(f"illustration for {child_name}")',
        'log.info("%s", request.child_name)',
        'print(child_name)',
        'log.warning("sending", extra={"child_name": value})',
        'span.set_attribute("child_name", value)',
        'mlflow.log_param("phone", phone_number)',
        'log.info("built %s", "x" + full_name)',
        'log.info("row %s", row["email"])',
    ],
)
def test_private_values_reaching_a_sink_are_findings(source):
    assert findings_for(source), source


def test_privacy_ok_suppresses_a_line():
    assert findings_for('log.info("page for %s", child_name)  # privacy: ok') == []


def test_non_sink_calls_are_ignored():
    """Only calls that send data somewhere are checked; ordinary code may pass PII around."""
    assert findings_for("page = render(child_name)") == []
    assert findings_for("name = record.child_name") == []


def test_literal_rules_still_apply(tmp_path):
    (tmp_path / "leaky.py").write_text('URL = f"https://api/books/{child_name}/pages"\n', encoding="utf-8")
    result = run_cli(tmp_path)
    assert "pii-in-url" in result.stdout
    assert "PRIVACY_RESULT: FAIL" in result.stdout


def test_hardcoded_secret_is_a_finding(tmp_path):
    (tmp_path / "config.py").write_text(
        'KEY = "AKIAIOSFODNN7EXAMPLE"\n', encoding="utf-8"
    )
    assert "hardcoded-secret" in run_cli(tmp_path).stdout


def test_typescript_still_uses_the_line_rule(tmp_path):
    (tmp_path / "ui.ts").write_text("console.log(childName);\n", encoding="utf-8")
    assert "pii-in-log" in run_cli(tmp_path).stdout


def test_test_files_are_skipped(tmp_path):
    (tmp_path / "test_pages.py").write_text('log.info("%s", child_name)\n', encoding="utf-8")
    nested = tmp_path / "tests"
    nested.mkdir()
    (nested / "helpers.py").write_text('log.info("%s", child_name)\n', encoding="utf-8")
    assert "PRIVACY_RESULT: PASS" in run_cli(tmp_path).stdout


# ------------------------------------------------- findings from the Copilot review of PR #11


def test_an_opt_in_elsewhere_in_the_file_does_not_excuse_a_write(tmp_path):
    """Consent on one code path is not consent on another."""
    (tmp_path / "store.py").write_text(
        "def settings():\n"
        "    save_audio = False\n"
        "    return save_audio\n"
        "\n"
        "def upload(blob, audio_bytes):\n"
        "    blob.upload_from_string(audio_bytes)\n",
        encoding="utf-8",
    )
    output = run_cli(tmp_path).stdout
    assert "audio-without-opt-in" in output, output


def test_an_opt_in_on_the_write_path_still_suppresses(tmp_path):
    """Only the audio rule is under test here; the object-name rule fires on `blob` regardless."""
    (tmp_path / "store.py").write_text(
        "def upload(blob, audio_bytes, save_audio):\n"
        "    if save_audio:\n"
        "        blob.upload_from_string(audio_bytes)\n",
        encoding="utf-8",
    )
    assert "audio-without-opt-in" not in run_cli(tmp_path).stdout


def test_the_repository_itself_passes():
    """The gate verify.py runs must pass on the tree, or every loop prompt stalls."""
    result = run_cli(REPO_ROOT / "src", REPO_ROOT / "app")
    assert "PRIVACY_RESULT: PASS" in result.stdout, result.stdout


def run_cli(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), *(str(p) for p in paths)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_exit_code_matches_the_result_line(tmp_path):
    (tmp_path / "clean.py").write_text('log.info("all good")\n', encoding="utf-8")
    clean = run_cli(tmp_path)
    assert clean.returncode == 0 and "PASS" in clean.stdout

    (tmp_path / "leaky.py").write_text('log.info("%s", child_name)\n', encoding="utf-8")
    dirty = run_cli(tmp_path)
    assert dirty.returncode == 1 and "FAIL" in dirty.stdout
