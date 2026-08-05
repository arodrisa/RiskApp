import os
import smtplib
from email.message import EmailMessage


class EmailDeliveryError(Exception):
    pass


def reset_delivery_mode():
    return os.getenv('APP_PASSWORD_RESET_DELIVERY', 'console').strip().lower()


def invitation_delivery_mode():
    return os.getenv('APP_INVITATION_DELIVERY', reset_delivery_mode()).strip().lower()


def _send_smtp(email: str, subject: str, content: str):
    username = os.getenv('SMTP_USERNAME', '').strip()
    password = os.getenv('SMTP_PASSWORD', '')
    sender = os.getenv('SMTP_FROM', username).strip()
    if not username or not password or not sender:
        raise EmailDeliveryError('SMTP_USERNAME, SMTP_PASSWORD, and SMTP_FROM must be configured')

    message = EmailMessage()
    message['Subject'] = subject
    message['From'] = sender
    message['To'] = email
    message.set_content(content)
    host = os.getenv('SMTP_HOST', 'smtp.gmail.com').strip()
    port = int(os.getenv('SMTP_PORT', '587'))
    try:
        with smtplib.SMTP(host, port, timeout=15) as client:
            client.starttls()
            client.login(username, password)
            client.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        raise EmailDeliveryError('Could not send email') from exc


def send_password_reset(email: str, reset_url: str):
    """Send a reset link through Gmail SMTP, or return it locally in development."""
    mode = reset_delivery_mode()
    if mode == 'console':
        return {'mode': 'console', 'reset_url': reset_url}
    if mode != 'smtp':
        raise EmailDeliveryError('Password-reset delivery is not configured')

    _send_smtp(email, 'Patrimonio password reset',
        'A password reset was requested for your Patrimonio account.\n\n'
        f'Open this link to set a new password:\n{reset_url}\n\n'
        'This link expires in 30 minutes. If you did not request it, you can ignore this email.',
    )
    return {'mode': 'smtp'}


def send_project_invitation(email: str, invite_url: str, role: str):
    mode = invitation_delivery_mode()
    if mode == 'console':
        return {'mode': 'console', 'invite_url': invite_url}
    if mode != 'smtp':
        raise EmailDeliveryError('Invitation delivery is not configured')
    _send_smtp(email, 'Patrimonio project invitation',
        'You have been invited to a Patrimonio project.\n\n'
        f'Role: {role}\n'
        f'Open this link to create your account:\n{invite_url}\n\n'
        'This link expires in 7 days. If you were not expecting this invitation, you can ignore this email.',
    )
    return {'mode': 'smtp'}
