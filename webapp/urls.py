from django.urls import path , include
from webapp.views import *

urlpatterns = [
path('', Home , name='Home'),
path('filedetails', Filedetails , name='Filedetails'),
path('delete', Deletedata, name='Deletedata'),
path('prediction', Prediction, name='Prediction'),




]
