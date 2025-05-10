# forms.py
from django import forms
from .models import Prediction

class PredictionForm(forms.ModelForm):
    class Meta:
        model = Prediction
        fields = ['gender', 'age', 'marital_status', 'anxiety', 'panic_attack', 'specialist_treatment', 'year_of_study']
