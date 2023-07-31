from rest_framework import serializers
from imagedetection.models import Serial_Detection


class Serial_DetectionSerializers(serializers.ModelSerializer):

    class Meta:
        model = Serial_Detection
        fields = [
            'id',
            'media',
        ]
