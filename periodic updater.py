import time
import psutil
from datetime import datetime
import sys
import logging
from logging.handlers import SMTPHandler
import smtplib

# --- Email Configuration ---
# Use environment variables for sensitive info (e.g., passwords) for security
SMTP_SERVER = "smtp.gmail.com"  # e.g., smtp.gmail.com, SMTP.office365.com
SMTP_PORT = 587  # typically 587 for TLS, 465 for SSL
SENDER_EMAIL = "richard.script.use@gmail.com"
RECIPIENT_EMAIL = "richardjiang2004@gmail.com"
EMAIL_PASSWORD = "uhau dvea emsk bair"  # Or an app-specific password

# --- Setup Logging with SMTPHandler ---
# Create a logger
error_logger = logging.getLogger(__name__)
error_logger.setLevel(logging.ERROR)

# Create the SMTP handler
try:
    smtp_handler = SMTPHandler(
        mailhost=(SMTP_SERVER, SMTP_PORT),
        fromaddr=SENDER_EMAIL,
        toaddrs=[RECIPIENT_EMAIL],
        subject="CRITICAL Error in Python Script",
        credentials=(SENDER_EMAIL, EMAIL_PASSWORD),
        secure=()  # Use secure=() for STARTTLS (port 587)
    )
    smtp_handler.setLevel(logging.ERROR)
    error_logger.addHandler(smtp_handler)
except smtplib.SMTPException as e:
    print(f"Failed to set up SMTP handler: {e}")


# --- Global Exception Handler Function ---
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """
    Logs unhandled exceptions and sends an email alert.
    """
    # Log the exception with full traceback
    error_logger.error("An unhandled exception occurred:",
                       exc_info=(exc_type, exc_value, exc_traceback))

    # Optionally, print to console as well (default behavior)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


# --- Add this helper somewhere near your email config (top of file) ---

def send_email(subject: str, body: str) -> None:
    """
    Send a plaintext email via SMTP (STARTTLS).
    Uses the global SMTP_* / SENDER_EMAIL / RECIPIENT_EMAIL / EMAIL_PASSWORD config.
    """
    msg = (
        f"From: {SENDER_EMAIL}\r\n"
        f"To: {RECIPIENT_EMAIL}\r\n"
        f"Subject: {subject}\r\n"
        f"\r\n"
        f"{body}\r\n"
    )

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, [RECIPIENT_EMAIL], msg.encode("utf-8", errors="replace"))


# Set the custom handler as the global exception hook
sys.excepthook = global_exception_handler

import collections


def get_last_n_lines(file_path, n=100):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Use deque to keep a running buffer of the last 'n' lines
        last_lines = collections.deque(file, n)
    return list(last_lines)


while any('Full_work_flow' in element for element in [p.cmdline() for p in psutil.process_iter() if p.name().lower() == "python.exe" and 'periodic updater.py' not in p.cmdline()[1]][0]):
    send_email(subject="Suite2p is still running",
               body=f"Suite2p is still running\ntime: {datetime.now()}" + "\n".join(get_last_n_lines('crosscorrelation.log', 100)))
    time.sleep(60 * 30)