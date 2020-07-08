from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    path('DataTable/', views.DataTable_view, name='DataTable'),
    path('TargetSelection/', views.TargetSelection, name='TargetSelection'),
    path('Exploratory/', views.Exploratory, name='Exploratory'),
    path('ExploratoryPlot/',views.ExploratoryPlot, name='ExploratoryPlot'),
    path('Report/',views.Report, name='Report'),
    path('ReportPlot/',views.Report_plot, name='ReportPlot')
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.DB_URL, document_root=settings.DB_ROOT)