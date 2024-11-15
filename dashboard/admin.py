from django.contrib import admin
from .models import *
#Register your models here.
admin.site.register(Notes)

admin.site.register(Homework)
admin.site.register(Todo)
admin.site.register(Coffee)

admin.site.register(Room)
admin.site.register(Message)
# Register your models here.
class contacting(admin.ModelAdmin):
  list_display=('name','email','message')
 
admin.site.register(Contact,contacting)