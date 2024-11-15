"""studentportal URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from re import template
from django.urls import include
from django.contrib import admin
from django.urls import path
from dashboard import views 
from django.contrib.auth import views as authview
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',include('dashboard.urls')),
    path('visualize/', views.visualize_data, name='visualize'),
    path("qabot/", views.qa_bot_view, name="qabot"),
    path('process/', views.process_complaint, name='process'),
    path('tabular/', views.csv_to_table, name='tabular'),





]
