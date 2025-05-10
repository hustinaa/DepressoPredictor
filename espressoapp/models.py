# models.py
from django.db import models

class Prediction(models.Model):
    id = models.BigAutoField(primary_key=True)
    GENDER_CHOICES = [(0, 'Male'), (1, 'Female')]
    MARITAL_CHOICES = [(0, 'Single'), (1, 'Married')]
    BINARY_CHOICES = [(0, 'No'), (1, 'Yes')]
    YEAR_CHOICES = [(1, 'Year 1'), (2, 'Year 2'), (3, 'Year 3'), (4, 'Year 4')]

    gender = models.IntegerField(choices=GENDER_CHOICES)
    age = models.IntegerField()
    marital_status = models.IntegerField(choices=MARITAL_CHOICES)
    anxiety = models.IntegerField(choices=BINARY_CHOICES)
    panic_attack = models.IntegerField(choices=BINARY_CHOICES)
    specialist_treatment = models.IntegerField(choices=BINARY_CHOICES)
    year_of_study = models.IntegerField(choices=YEAR_CHOICES)
    knn_prediction = models.IntegerField(null=True, blank=True)
    decision_tree_prediction = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"Prediction {self.id}"
