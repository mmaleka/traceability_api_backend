from django.urls import path
from . import views
from django.contrib.auth.decorators import login_required

app_name = 'api-traceability'

urlpatterns = [

     path('traceability_list/', 
          views.traceabilityList.as_view(), 
          name='traceabilityList'),

     path('traceability_post/', 
          views.traceabilityPost.as_view(), 
          name='traceabilityPost'),
          
     path('ScannerDataSerilisersAPIView',
          views.ScannerDataSerilisersAPIView.as_view(),
          name='ScannerDataSerilisersAPIView'),
    
]

