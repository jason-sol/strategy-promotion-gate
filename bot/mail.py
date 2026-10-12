"""Gmail SMTP transport.

The one place the bot sends mail from. Reports, alerts and command replies are
all built elsewhere and handed here, so no module has to depend on another's
subject matter to reach the mailbox.

Requires env vars: GMAIL_ADDRESS, GMAIL_APP_PASSWORD (a Google App Password —
normal account passwords are rejected by Gmail SMTP), optional BRIEFING_TO.
"""

import os
import smtplib
from email.message import EmailMessage


def mail_configured() -> bool:
    return bool(os.getenv("GMAIL_ADDRESS") and os.getenv("GMAIL_APP_PASSWORD"))


def send_gmail(subject: str, body: str) -> None:
    address = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    if not address or not password:
        raise RuntimeError("GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set")
    msg = EmailMessage()
    msg["From"] = address
    msg["To"] = os.getenv("BRIEFING_TO", address)
    msg["Subject"] = subject
    msg.set_content(body + "\n\n-- strategy-promotion-gate (paper account)")
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.starttls()
        server.login(address, password)
        server.send_message(msg)
