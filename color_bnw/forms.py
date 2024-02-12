from django import forms

from color_bnw.models import Color


class ColorForm(forms.ModelForm):

    def __init__(self, *args, **kwargs):
        super(ColorForm, self).__init__(*args, **kwargs)
        for field in self.visible_fields():
            print(field)
            field.field.widget.attrs["class"] = "form-control"

    class Meta:
        model = Color
        fields = ["image"]
