from django.urls import path, re_path
from django.views.static import serve

from ragplay_web import views
from ragplay_web.settings import BASE_DIR

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("attention", views.attention, name="attention"),
    path("ask", views.ask, name="ask"),
    path("ask/send", views.ask_send, name="ask_send"),
    path("ask/new", views.ask_new, name="ask_new"),
    path("history", views.history, name="history"),
    path("health", views.health, name="health"),
    # runserver serves static files only with DEBUG on; this is a single-user localhost app
    re_path(r"^static/(?P<path>.*)$", serve, {"document_root": BASE_DIR / "static"}),
]
