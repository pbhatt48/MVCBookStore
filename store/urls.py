from django.conf.urls import url
#from MasterPythonClass.MVCBOOKSTORESITE.Development.bookstore.store import views
from . import views

urlpatterns = [
    url(r'^$', views.store, name='index'),
    #url(r'^store/', 'store.views.store', name='store'),

]
