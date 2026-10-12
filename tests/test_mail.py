"""Gmail transport: the SMTP send and the credential gate that guards it."""

from unittest.mock import patch

from bot.mail import mail_configured, send_gmail


class TestMailConfigured:
    def test_true_only_with_both_credentials(self, monkeypatch):
        monkeypatch.setenv("GMAIL_ADDRESS", "me@gmail.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")
        assert mail_configured() is True

    def test_false_without_password(self, monkeypatch):
        monkeypatch.setenv("GMAIL_ADDRESS", "me@gmail.com")
        monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
        assert mail_configured() is False

    def test_false_without_address(self, monkeypatch):
        monkeypatch.delenv("GMAIL_ADDRESS", raising=False)
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")
        assert mail_configured() is False


class TestSendGmail:
    def test_sends_via_smtp_starttls(self, monkeypatch):
        monkeypatch.setenv("GMAIL_ADDRESS", "me@gmail.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")
        monkeypatch.delenv("BRIEFING_TO", raising=False)
        with patch("bot.mail.smtplib.SMTP") as smtp_cls:
            server = smtp_cls.return_value.__enter__.return_value
            send_gmail("subject", "body")
            smtp_cls.assert_called_once_with("smtp.gmail.com", 587, timeout=30)
            server.starttls.assert_called_once()
            server.login.assert_called_once_with("me@gmail.com", "abcd efgh ijkl mnop")
            msg = server.send_message.call_args.args[0]
            assert msg["To"] == "me@gmail.com"
            assert msg["Subject"] == "subject"

    def test_raises_without_credentials(self, monkeypatch):
        monkeypatch.delenv("GMAIL_ADDRESS", raising=False)
        monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
        try:
            send_gmail("s", "b")
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
