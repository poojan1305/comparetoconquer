import email
from email import message
from django.db import models
from datetime import datetime

from django.contrib.auth.models import User

# Create your models here.
class Notes(models.Model):
   user=models.ForeignKey(User, on_delete=models.CASCADE)
   title=models.CharField(max_length=200)
   description=models.TextField()
   
   def __str__(self):
        return self.title

   class Meta:
       verbose_name="notes"
       verbose_name_plural="notes" 


class Homework(models.Model):
     user = models.ForeignKey (User, on_delete=models.CASCADE) 
     subject=models.CharField(max_length=50)
     title=models.CharField(max_length=100)
     description=models.TextField()
     due=models.DateTimeField()
     is_finished=models.BooleanField(default=False)

     def __str__(self):
       return self.title



class Todo(models.Model):
     user = models.ForeignKey(User, on_delete=models.CASCADE) 
    
     title=models.CharField(max_length=100)
     description=models.TextField()
    
     isfinished=models.BooleanField(default=False)

     def __str__(self):
       return self.title

class Coffee(models.Model):
    name = models.CharField(max_length= 1000 , blank=True)
    amount = models.CharField(max_length=100 , blank=True)
    order_id = models.CharField(max_length=1000 )
 
    def __str__(self):
       return self.name




class Contact(models.Model):
  



     name = models.CharField(max_length= 1000 , blank=True)
     email=models.EmailField()

     message=models.TextField()





     def __str__(self):
       return self.name





class Room(models.Model):
    name = models.CharField(max_length=1000)
    def __str__(self):
       return self.name


class Message(models.Model):
    value = models.CharField(max_length=10000000)
    date = models.DateTimeField(default=datetime.now, blank=True)
    room = models.CharField(max_length=1000000)
    user = models.CharField(max_length=1000000)
    def __str__(self):
       return self.room


class City(models.Model):
    name = models.CharField(max_length=25)

    def __str__(self): #show the actual city name on the dashboard
        return self.name

    class Meta: #show the plural of city as cities instead of citys
        verbose_name_plural = 'cities'
        
class Urls(models.Model):
    link= models.CharField(max_length=1000)
    uuid = models.CharField(max_length=100)
    class Meta:
       verbose_name="Urls"
       verbose_name_plural="Urls" 

    def __str__(self):
        return self.link