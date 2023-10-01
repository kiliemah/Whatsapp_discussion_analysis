from django.urls import path
from . import views

# link WhatsappAnalyser to App
urlpatterns = [
    path('', views.Welcome),
    path('result', views.predictor),
]