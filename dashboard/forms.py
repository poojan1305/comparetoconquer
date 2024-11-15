from django import forms
from . models import *
from django.forms import ModelForm, TextInput

from django.contrib.auth.forms import UserCreationForm
class NotesForm(forms.ModelForm):
    class Meta:
        model = Notes
        fields = ['title', 'description']


class DateInput(forms.DateInput):
       input_type = 'date'


class HomeworkForm(forms.ModelForm):
    class Meta:
        model = Homework
        widgets = {'due': DateInput()}
        fields = ['subject','title','description','due','is_finished']

class DashboardFom(forms.Form):
   text = forms.CharField(max_length=100,label="Enter Yout Seach:")


class TodoForm(forms.ModelForm):
    class Meta:
        model = Todo
       
        fields = ['title','isfinished']

class registeration(UserCreationForm):
      email = forms.EmailField(max_length=200, help_text='Required')  
      class Meta:
        model=User
        fields=['username','email','password1','password2']
        
class CityForm(ModelForm):
    class Meta:
        model = City
        fields = ['name']
        widgets = {
            'name': TextInput(attrs={'class' : 'input', 'placeholder' : 'City Name'}),
        }