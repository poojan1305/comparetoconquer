from unicodedata import name
from django.urls import path
from  . import views
urlpatterns = [
    path('', views.home, name="home"),



    path('contact', views.contact, name="contact"),


   path('faq', views.faq, name="faq"),


     path('aboutus', views.about, name="aboutus"),

    path('success', views.success, name="success"),
     path('about', views.about, name="about"),
       path('qabot', views.qa_bot_view, name="qabot"),
         path('process', views.process_complaint, name="process"),
             path('visualize/', views.visualize_data, name='visualize'),
                 path('tabular/', views.csv_to_table, name='tabular'),







]
