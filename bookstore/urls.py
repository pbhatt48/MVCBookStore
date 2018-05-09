from django.conf.urls import include, url
from django.contrib import admin
#from django.core.urlresolvers import reverse

#from MasterPythonClass.MVCBOOKSTORESITE.Development.bookstore import store

urlpatterns = [
    # Examples:
    # url(r'^$', 'bookstore.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
    url(r'^store/', include('store.urls'), name='store'),
    #url(r'^admin/', include('admin.site.urls'), name='index'),
    #url(r'^store/', 'store.views.store', name='store'),
    url(r'^accounts/', include('registration.backends.default.urls')),
    url(r'^admin/', include(admin.site.urls)),
]
