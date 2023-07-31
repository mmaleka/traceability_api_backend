from django.db import models

# Create your models here.

class Serial_Detection(models.Model):
    media = models.FileField()
    created_on = models.DateTimeField(auto_now_add=True)


    class Meta:
        ordering = ['-created_on']

    def __str__(self):
        return self.pk


