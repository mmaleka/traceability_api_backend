from rest_framework import serializers
from traceability.models import traceability, ScannerData




class traceabilitySerializers(serializers.ModelSerializer):

    class Meta:
        model = traceability
        fields = '__all__'



class ScannerDataDataSerialisersv1(serializers.ModelSerializer):

    class Meta:
        model=ScannerData
        fields = [
            "id",
            "shell_no",
            "cast_code",
            "heat_code",
            "shell_serial_no",
            "location",
            "added",
            "created_at"
        ]


