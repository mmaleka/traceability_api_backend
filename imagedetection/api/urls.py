from django.conf.urls import url
from django.urls import path, include
from . import views

app_name = 'api-serial_detection_api'


urlpatterns = [

    path('', views.Serial_DetectionAPIView.as_view(), name="serial_detection_api"),

]

