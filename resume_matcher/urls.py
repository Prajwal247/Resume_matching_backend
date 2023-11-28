from django.urls import path
from . import views

urlpatterns = [
    path('matcher/', views.matcher, name='matcher'),
    # Define more URL patterns as needed
]