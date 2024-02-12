from django.db import models


# Create your models here.
class Color(models.Model):
    image = models.ImageField(upload_to="colored")


class Product(models.Model):
    img_colorized = models.ImageField(upload_to="colored/finished/")
