from django.contrib import admin
from django.urls import path
from .views import render



  # Import your view
urlpatterns = [
    path('data/<path:text>/', render, name='render'),
]

