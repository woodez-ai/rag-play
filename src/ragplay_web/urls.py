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
    path("spending", views.spending, name="spending"),
    path("insights", views.insights, name="insights"),
    path("insights/review/run", views.review_run, name="review_run"),
    path("insights/<slug:tab>", views.insights, name="insights_tab"),
    path("budgets", views.budgets_page, name="budgets"),
    path("budgets/set", views.budget_set, name="budget_set"),
    path("budgets/remove", views.budget_remove, name="budget_remove"),
    path("categories", views.categories, name="categories"),
    path("categories/set", views.category_set, name="categories_set"),
    path("categories/review", views.category_review, name="categories_review"),
    path("categories/review/answer", views.category_review_answer, name="categories_answer"),
    path("import", views.import_page, name="import"),
    path("import/upload", views.import_upload, name="import_upload"),
    path("import/job/<slug:job_id>", views.import_job, name="import_job"),
    path("history", views.history, name="history"),
    path("health", views.health, name="health"),
    # runserver serves static files only with DEBUG on; this is a single-user localhost app
    re_path(r"^static/(?P<path>.*)$", serve, {"document_root": BASE_DIR / "static"}),
]
