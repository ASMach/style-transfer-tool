from django.db import models

# Create your models here.


class Image(models.Model):
    filename = models.CharField(max_length=200)
    pub_date = models.DateTimeField("date published")
