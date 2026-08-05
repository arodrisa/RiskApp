import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import datetime, timedelta

from fastapi import HTTPException, Request, Response
from sqlalchemy import func

from app.database import SessionLocal
from app import models
from app import email_delivery


SESSION_COOKIE = 'patrimonio_session'
CSRF_COOKIE = 'patrimonio_csrf'
_LOGIN_ATTEMPTS = {}


def auth_enabled():
    return os.getenv('APP_AUTH_ENABLED', 'false').strip().lower() in {'1', 'true', 'yes', 'on'}


def env_flag(name: str, default: bool = False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def auth_username():
    return os.getenv('PATRIMONIO_USERNAME', 'admin')


def auth_password():
    return os.getenv('PATRIMONIO_PASSWORD', '')


def session_secret():
    return os.getenv('PATRIMONIO_SESSION_SECRET') or auth_password() or 'dev-session-secret'


def app_environment():
    return os.getenv('APP_ENV', 'development').strip().lower()


def is_production_environment():
    return app_environment() in {'production', 'prod'}


def validate_production_settings():
    if not is_production_environment():
        return
    secret = session_secret()
    if len(secret) < 32 or 'change-me' in secret.lower() or secret == 'dev-session-secret':
        raise RuntimeError('Production requires a strong PATRIMONIO_SESSION_SECRET')
    if not auth_enabled():
        raise RuntimeError('Production requires APP_AUTH_ENABLED=true')
    if not cookie_secure():
        raise RuntimeError('Production requires APP_COOKIE_SECURE=true')
    public_url = os.getenv('APP_PUBLIC_URL', '').strip().lower()
    if not public_url.startswith('https://'):
        raise RuntimeError('Production requires an HTTPS APP_PUBLIC_URL')


def bootstrap_token():
    return os.getenv('APP_BOOTSTRAP_TOKEN', '')


def normalize_email(value: str):
    return str(value or '').strip().lower()


def password_hash(password: str):
    """Use stdlib scrypt so the beta does not need a system crypt dependency."""
    if len(password or '') < 8:
        raise HTTPException(status_code=400, detail='Password must contain at least 8 characters')
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode('utf-8'), salt=salt, n=16384, r=8, p=1)
    return 'scrypt$16384$8$1$%s$%s' % (_b64_encode(salt), _b64_encode(digest))


def verify_password(password: str, stored_hash: str):
    try:
        algorithm, n, r, p, salt, digest = stored_hash.split('$', 5)
        if algorithm != 'scrypt':
            return False
        candidate = hashlib.scrypt(
            password.encode('utf-8'),
            salt=_b64_decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
        )
        return hmac.compare_digest(candidate, _b64_decode(digest))
    except (TypeError, ValueError):
        return False


def session_ttl_seconds():
    return int(os.getenv('PATRIMONIO_SESSION_TTL_SECONDS', '43200'))


def password_reset_ttl_seconds():
    return int(os.getenv('APP_PASSWORD_RESET_TTL_SECONDS', '1800'))


def cookie_secure():
    return env_flag('APP_COOKIE_SECURE', False)


def cookie_samesite():
    value = os.getenv('APP_COOKIE_SAMESITE', 'strict').strip().lower()
    return value if value in {'strict', 'lax', 'none'} else 'strict'


def login_rate_limit_attempts():
    return int(os.getenv('APP_LOGIN_RATE_LIMIT_ATTEMPTS', '5'))


def login_rate_limit_window_seconds():
    return int(os.getenv('APP_LOGIN_RATE_LIMIT_WINDOW_SECONDS', '300'))


def restore_enabled():
    return env_flag('APP_RESTORE_ENABLED', not auth_enabled())


def _b64_encode(data: bytes):
    return base64.urlsafe_b64encode(data).decode('ascii').rstrip('=')


def _b64_decode(data: str):
    padding = '=' * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode('ascii'))


def _sign(payload: str):
    return hmac.new(session_secret().encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()


def create_session_token(username: str, user_id: int = None, session_version: int = None):
    data = {
        'sub': username,
        'iat': int(time.time()),
        'nonce': secrets.token_urlsafe(12),
    }
    if user_id is not None:
        data['uid'] = user_id
        data['sv'] = session_version
    payload = _b64_encode(json.dumps(data, separators=(',', ':')).encode('utf-8'))
    return f'{payload}.{_sign(payload)}'


def create_csrf_token():
    value = secrets.token_urlsafe(24)
    return f'{value}.{_sign(value)}'


def verify_csrf_token(token: str):
    if not token or '.' not in token:
        return False
    value, signature = token.rsplit('.', 1)
    return hmac.compare_digest(_sign(value), signature)


def _decode_session_token(token: str):
    if not token or '.' not in token:
        return None
    payload, signature = token.rsplit('.', 1)
    if not hmac.compare_digest(_sign(payload), signature):
        return None
    try:
        data = json.loads(_b64_decode(payload))
    except Exception:
        return None
    if int(time.time()) - int(data.get('iat') or 0) > session_ttl_seconds():
        return None
    return data


def verify_session_token(token: str):
    data = _decode_session_token(token)
    return data.get('sub') if data else None


def _database_user_count(db=None):
    owns_session = db is None
    db = db or SessionLocal()
    try:
        return db.query(models.User).count()
    except Exception:
        return 0
    finally:
        if owns_session:
            db.close()


def _session_context(request: Request, db=None):
    if not auth_enabled():
        return {'username': 'api', 'role': 'owner', 'project_name': 'Development'}
    token_data = _decode_session_token(request.cookies.get(SESSION_COOKIE, ''))
    if not token_data:
        return None
    if not _database_user_count(db):
        username = token_data.get('sub')
        return {'username': username, 'role': 'owner', 'project_name': 'Legacy'} if username else None

    owns_session = db is None
    db = db or SessionLocal()
    try:
        user_id = token_data.get('uid')
        if not user_id:
            return None
        user = db.query(models.User).filter(models.User.id == user_id, models.User.is_active.is_(True)).first()
        if user is None or token_data.get('sv') != user.session_version:
            return None
        membership = db.query(models.ProjectMembership).filter(
            models.ProjectMembership.user_id == user.id,
        ).order_by(models.ProjectMembership.id).first()
        if membership is None:
            return None
        return {
            'username': user.email,
            'user_id': user.id,
            'role': membership.role,
            'project_id': membership.project_id,
            'project_name': membership.project.name,
        }
    finally:
        if owns_session:
            db.close()


def current_actor(request: Request):
    context = _session_context(request)
    return context['username'] if context else 'anonymous'


def require_auth(request: Request):
    context = _session_context(request)
    if not context:
        raise HTTPException(status_code=401, detail='Authentication required')
    return context['username']


def require_project_admin(request: Request):
    context = _session_context(request)
    if not context:
        raise HTTPException(status_code=401, detail='Authentication required')
    if context.get('role') not in {'owner', 'admin'}:
        raise HTTPException(status_code=403, detail='Project administrator access is required')
    return context['username']


def require_project_editor(request: Request):
    context = _session_context(request)
    if not context:
        raise HTTPException(status_code=401, detail='Authentication required')
    if context.get('role') not in {'owner', 'admin', 'editor'}:
        raise HTTPException(status_code=403, detail='Project editor access is required')
    return context['username']


def status(request: Request, db=None):
    enabled = auth_enabled()
    context = _session_context(request, db)
    authenticated = not enabled or context is not None
    return {
        'enabled': enabled,
        'authenticated': authenticated,
        'username': context['username'] if context else None,
        'csrf_token': csrf_token_for_authenticated_request(request) if authenticated else None,
        'restore_enabled': restore_enabled(),
        'needs_bootstrap': enabled and not _database_user_count(db),
        'role': context.get('role') if context else None,
        'project_name': context.get('project_name') if context else None,
    }


def _client_key(request: Request, username: str):
    client_host = request.client.host if request.client else 'unknown'
    return f'{client_host}:{username}'


def _check_login_rate_limit(request: Request, username: str):
    now = time.time()
    window = login_rate_limit_window_seconds()
    attempts = login_rate_limit_attempts()
    key = _client_key(request, username)
    recent = [stamp for stamp in _LOGIN_ATTEMPTS.get(key, []) if now - stamp < window]
    _LOGIN_ATTEMPTS[key] = recent
    if len(recent) >= attempts:
        raise HTTPException(status_code=429, detail='Too many login attempts. Try again later.')


def _record_failed_login(request: Request, username: str):
    key = _client_key(request, username)
    _LOGIN_ATTEMPTS.setdefault(key, []).append(time.time())


def _clear_login_attempts(request: Request, username: str):
    _LOGIN_ATTEMPTS.pop(_client_key(request, username), None)


def _complete_login(response: Response, username: str, user_id: int = None, session_version: int = None, role: str = None, project_name: str = None):
    csrf_token = create_csrf_token()
    response.set_cookie(
        SESSION_COOKIE,
        create_session_token(username, user_id=user_id, session_version=session_version),
        httponly=True,
        samesite=cookie_samesite(),
        secure=cookie_secure(),
        max_age=session_ttl_seconds(),
        path='/',
    )
    response.set_cookie(
        CSRF_COOKIE,
        csrf_token,
        httponly=False,
        samesite=cookie_samesite(),
        secure=cookie_secure(),
        max_age=session_ttl_seconds(),
        path='/',
    )
    return {
        'authenticated': True,
        'username': username,
        'csrf_token': csrf_token,
        'restore_enabled': restore_enabled(),
        'role': role,
        'project_name': project_name,
    }


def login(request: Request, response: Response, username: str, password: str, db=None):
    if not auth_enabled():
        return {'authenticated': True, 'username': 'dev', 'csrf_token': None, 'restore_enabled': restore_enabled(), 'role': 'owner', 'project_name': 'Development'}
    if _database_user_count(db):
        email = normalize_email(username)
        _check_login_rate_limit(request, email)
        user = db.query(models.User).filter(models.User.email == email, models.User.is_active.is_(True)).first()
        if user is None or not verify_password(password, user.password_hash):
            _record_failed_login(request, email)
            raise HTTPException(status_code=401, detail='Invalid email or password')
        membership = db.query(models.ProjectMembership).filter(
            models.ProjectMembership.user_id == user.id,
        ).order_by(models.ProjectMembership.id).first()
        if membership is None:
            raise HTTPException(status_code=403, detail='This account does not have a project membership')
        user.last_login_at = datetime.utcnow()
        db.add(user)
        db.commit()
        _clear_login_attempts(request, email)
        return _complete_login(
            response,
            user.email,
            user_id=user.id,
            session_version=user.session_version,
            role=membership.role,
            project_name=membership.project.name,
        )
    expected_username = auth_username()
    expected_password = auth_password()
    if not expected_password:
        raise HTTPException(status_code=500, detail='Authentication is enabled but PATRIMONIO_PASSWORD is not set')
    _check_login_rate_limit(request, username)
    if not hmac.compare_digest(username, expected_username) or not hmac.compare_digest(password, expected_password):
        _record_failed_login(request, username)
        raise HTTPException(status_code=401, detail='Invalid username or password')
    _clear_login_attempts(request, username)
    return _complete_login(response, username, role='owner', project_name='Legacy')


def bootstrap(request: Request, response: Response, payload, db):
    if _database_user_count(db):
        raise HTTPException(status_code=409, detail='The first account has already been created')
    configured_token = bootstrap_token()
    if app_environment() in {'production', 'prod'} and not configured_token:
        raise HTTPException(status_code=503, detail='Production bootstrap requires APP_BOOTSTRAP_TOKEN')
    if configured_token and not hmac.compare_digest(configured_token, payload.setup_token or ''):
        raise HTTPException(status_code=403, detail='Bootstrap token is invalid')
    email = normalize_email(payload.email)
    if not email or '@' not in email:
        raise HTTPException(status_code=400, detail='A valid email address is required')
    owner = None
    if payload.person_owner_id is not None:
        owner = db.query(models.Owner).filter(models.Owner.id == payload.person_owner_id).first()
        if owner is None or owner.type != 'person':
            raise HTTPException(status_code=400, detail='Select an existing person entity')
    from app import crud
    project = crud.initialize_project_data(db)
    user = models.User(
        email=email,
        display_name=payload.display_name.strip() or email,
        password_hash=password_hash(payload.password),
        person_owner_id=owner.id if owner else None,
    )
    db.add(user)
    db.flush()
    membership = models.ProjectMembership(project_id=project.id, user_id=user.id, role='owner')
    db.add(membership)
    db.commit()
    return _complete_login(
        response,
        user.email,
        user_id=user.id,
        session_version=user.session_version,
        role=membership.role,
        project_name=project.name,
    )


def create_invitation(request: Request, payload, db):
    context = _session_context(request, db)
    if not context or context.get('role') not in {'owner', 'admin'}:
        raise HTTPException(status_code=403, detail='Project administrator access is required')
    email = normalize_email(payload.email)
    if not email or '@' not in email:
        raise HTTPException(status_code=400, detail='A valid email address is required')
    if db.query(models.ProjectMembership).join(models.User).filter(
        models.ProjectMembership.project_id == context['project_id'],
        func.lower(models.User.email) == email,
    ).first():
        raise HTTPException(status_code=409, detail='This email already belongs to the project')
    db.query(models.ProjectInvitation).filter(
        models.ProjectInvitation.project_id == context['project_id'],
        func.lower(models.ProjectInvitation.email) == email,
        models.ProjectInvitation.accepted_at.is_(None),
    ).delete(synchronize_session=False)
    raw_token = secrets.token_urlsafe(32)
    invitation = models.ProjectInvitation(
        project_id=context['project_id'],
        email=email,
        role=payload.role,
        token_hash=hashlib.sha256(raw_token.encode('utf-8')).hexdigest(),
        invited_by_user_id=context['user_id'],
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    db.add(invitation)
    public_url = os.getenv('APP_PUBLIC_URL', '').rstrip('/')
    invite_url = f'{public_url}/?invite={raw_token}' if public_url else f'/?invite={raw_token}'
    try:
        delivery = email_delivery.send_project_invitation(email, invite_url, payload.role)
        db.commit()
    except email_delivery.EmailDeliveryError as exc:
        db.rollback()
        if app_environment() in {'production', 'prod'}:
            raise HTTPException(status_code=503, detail='Invitation email delivery is unavailable') from exc
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    db.refresh(invitation)
    return invitation, delivery.get('invite_url') if delivery.get('mode') == 'console' else None


def update_project_user(request: Request, user_id: int, payload, db):
    context = _session_context(request, db)
    if not context or context.get('role') not in {'owner', 'admin'}:
        raise HTTPException(status_code=403, detail='Project administrator access is required')
    membership = db.query(models.ProjectMembership).filter(
        models.ProjectMembership.project_id == context['project_id'],
        models.ProjectMembership.user_id == user_id,
    ).first()
    if membership is None:
        raise HTTPException(status_code=404, detail='Project user not found')
    if membership.role == 'owner':
        raise HTTPException(status_code=400, detail='The project owner cannot be changed or deactivated')
    data = payload.dict(exclude_unset=True)
    if 'role' in data:
        membership.role = data['role']
    if 'is_active' in data:
        membership.user.is_active = bool(data['is_active'])
        membership.user.session_version += 1
    db.add(membership)
    db.add(membership.user)
    db.commit()
    db.refresh(membership)
    return membership


def accept_invitation(request: Request, response: Response, payload, db):
    token_hash = hashlib.sha256((payload.token or '').encode('utf-8')).hexdigest()
    invitation = db.query(models.ProjectInvitation).filter(
        models.ProjectInvitation.token_hash == token_hash,
        models.ProjectInvitation.accepted_at.is_(None),
    ).first()
    if invitation is None or invitation.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail='This invitation is invalid or has expired')
    if db.query(models.User).filter(models.User.email == invitation.email).first():
        raise HTTPException(status_code=409, detail='An account already exists for this email. Sign in before joining another project.')
    owner = None
    if payload.person_owner_id is not None:
        owner = db.query(models.Owner).filter(models.Owner.id == payload.person_owner_id).first()
        if owner is None or owner.type != 'person' or owner.project_id != invitation.project_id:
            raise HTTPException(status_code=400, detail='Select a person entity in this project')
    user = models.User(
        email=invitation.email,
        display_name=payload.display_name.strip() or invitation.email,
        password_hash=password_hash(payload.password),
        person_owner_id=owner.id if owner else None,
    )
    db.add(user)
    db.flush()
    membership = models.ProjectMembership(project_id=invitation.project_id, user_id=user.id, role=invitation.role)
    invitation.accepted_at = datetime.utcnow()
    db.add(membership)
    db.add(invitation)
    db.commit()
    return _complete_login(
        response,
        user.email,
        user_id=user.id,
        session_version=user.session_version,
        role=membership.role,
        project_name=invitation.project.name,
    )


def request_password_reset(payload, db):
    message = 'If an active account matches that email address, a reset link has been sent.'
    email = normalize_email(payload.email)
    user = db.query(models.User).filter(
        models.User.email == email,
        models.User.is_active.is_(True),
    ).first()
    if user is None:
        return {'message': message, 'dev_reset_url': None}

    raw_token = secrets.token_urlsafe(32)
    reset_url = f"{os.getenv('APP_PUBLIC_URL', '').rstrip('/')}/?reset={raw_token}"
    if not reset_url.startswith('http'):
        reset_url = f'/?reset={raw_token}'
    db.query(models.PasswordResetToken).filter(
        models.PasswordResetToken.user_id == user.id,
        models.PasswordResetToken.used_at.is_(None),
    ).delete(synchronize_session=False)
    token = models.PasswordResetToken(
        user_id=user.id,
        token_hash=hashlib.sha256(raw_token.encode('utf-8')).hexdigest(),
        expires_at=datetime.utcnow() + timedelta(seconds=password_reset_ttl_seconds()),
    )
    db.add(token)
    try:
        delivery = email_delivery.send_password_reset(user.email, reset_url)
        db.commit()
    except email_delivery.EmailDeliveryError as exc:
        db.rollback()
        if app_environment() in {'production', 'prod'}:
            raise HTTPException(status_code=503, detail='Password-reset email delivery is unavailable') from exc
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        'message': message,
        'dev_reset_url': delivery.get('reset_url') if delivery.get('mode') == 'console' else None,
    }


def confirm_password_reset(payload, db):
    token_hash = hashlib.sha256((payload.token or '').encode('utf-8')).hexdigest()
    token = db.query(models.PasswordResetToken).filter(
        models.PasswordResetToken.token_hash == token_hash,
        models.PasswordResetToken.used_at.is_(None),
    ).first()
    if token is None or token.expires_at < datetime.utcnow() or not token.user.is_active:
        raise HTTPException(status_code=400, detail='This password reset link is invalid or has expired')
    token.user.password_hash = password_hash(payload.password)
    token.user.session_version += 1
    token.used_at = datetime.utcnow()
    db.add(token.user)
    db.add(token)
    db.commit()
    return {'message': 'Password updated. You can now sign in.'}


def csrf_token_for_authenticated_request(request: Request):
    if not auth_enabled():
        return None
    if not _session_context(request):
        return None
    token = request.cookies.get(CSRF_COOKIE, '')
    return token if verify_csrf_token(token) else None


def require_csrf(request: Request):
    if not auth_enabled():
        return None
    cookie_token = request.cookies.get(CSRF_COOKIE, '')
    header_token = request.headers.get('x-csrf-token', '')
    if not verify_csrf_token(cookie_token) or not hmac.compare_digest(cookie_token, header_token):
        raise HTTPException(status_code=403, detail='CSRF token missing or invalid')
    return None


def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, path='/')
    response.delete_cookie(CSRF_COOKIE, path='/')
    return {'authenticated': False, 'username': None, 'csrf_token': None, 'restore_enabled': restore_enabled()}
