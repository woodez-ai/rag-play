"""Django settings for the local web UI (`rag web`).

Local only: binds to 127.0.0.1 and accepts only localhost Host headers. There is no login, so
never expose it on the network. Django keeps no database of its own: pages use the ragplay
psycopg connections (read-only for questions), and sessions are files under data/web/.
"""

import os
import secrets
from pathlib import Path

from django.utils.csp import CSP

from ragplay import db

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = db.ROOT / "data/web"  # gitignored (data/)


def _secret_key() -> str:
    """DJANGO_SECRET_KEY from the environment / .env, else a random key generated once and kept in
    data/web/secret_key (mode 600). Only signs session cookies for this machine."""
    if os.environ.get("DJANGO_SECRET_KEY"):
        return os.environ["DJANGO_SECRET_KEY"]
    path = DATA_DIR / "secret_key"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600)
        path.write_text(secrets.token_urlsafe(50))
    return path.read_text().strip()


SECRET_KEY = _secret_key()
DEBUG = os.environ.get("RAGPLAY_WEB_DEBUG") == "1"
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]

# which ragplay schema the pages read (`rag web --schema ragplay_sample` for synthetic data)
RAG_SCHEMA = os.environ.get("RAGPLAY_WEB_SCHEMA", db.SCHEMA)
# uploaded statements: the real ones join cc_statements/ (gitignored); synthetic ones stay apart
IMPORT_DIR = db.ROOT / "cc_statements" if RAG_SCHEMA == db.SCHEMA else DATA_DIR / "uploads" / RAG_SCHEMA
MAX_PDF_BYTES = 25 * 1024 * 1024
DATA_UPLOAD_MAX_NUMBER_FILES = 60

INSTALLED_APPS = [
    "django.contrib.sessions",
    "django.contrib.staticfiles",
    "ragplay_web",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.csp.ContentSecurityPolicyMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
ROOT_URLCONF = "ragplay_web.urls"
WSGI_APPLICATION = "ragplay_web.wsgi.application"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",
        "django.template.context_processors.csrf",
        "ragplay_web.views.nav",
    ]},
}]

DATABASES: dict = {}
SESSION_ENGINE = "django.contrib.sessions.backends.file"
SESSION_FILE_PATH = str(DATA_DIR / "sessions")
SESSION_COOKIE_AGE = 60 * 60 * 24 * 7
SESSION_SERIALIZER = "django.contrib.sessions.serializers.JSONSerializer"
Path(SESSION_FILE_PATH).mkdir(parents=True, exist_ok=True)

# No inline scripts and no third-party hosts: htmx and Chart.js are vendored in static/vendor.
# Inline styles are allowed (bar widths, Chart.js canvas sizing).
SECURE_CSP = {
    "default-src": [CSP.SELF],
    "script-src": [CSP.SELF],
    "style-src": [CSP.SELF, CSP.UNSAFE_INLINE],
    "img-src": [CSP.SELF, "data:"],
    "connect-src": [CSP.SELF],
    "frame-ancestors": [CSP.NONE],
    "form-action": [CSP.SELF],
}
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"

STATIC_URL = "static/"
USE_TZ = True
TIME_ZONE = "America/Toronto"
LANGUAGE_CODE = "en-us"
