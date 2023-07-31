from rest_framework import generics, mixins
from rest_framework.views import APIView
import datetime

import requests



from traceability.models import traceability, ScannerData
from . serializers import traceabilitySerializers, ScannerDataDataSerialisersv1


class traceabilityList(generics.ListAPIView):
    queryset = traceability.objects.all()
    serializer_class = traceabilitySerializers


class traceabilityPost(generics.CreateAPIView):
    serializer_class = traceabilitySerializers



class ScannerDataSerilisersAPIView(
    mixins.CreateModelMixin, 
    generics.ListAPIView,
    ):


    lookup_field = 'id'
    serializer_class =  ScannerDataDataSerialisersv1

    def post_server(self, shell_no, cast_code, heat_code, shell_serial_no, location):
        print("adding data to http://192.168.2.101/api-hardnessReport/ScannerDataSerilisersAPIView server")
        '''Add the data to the svr062 server'''
        url = r'http://192.168.2.101/api-hardnessReport/ScannerDataSerilisersAPIView'
        pload = {
            'shell_no': shell_no,
            'cast_code': cast_code, 
            'heat_code': heat_code,
            'shell_serial_no': shell_serial_no,
            'location': location,
            }
        x = requests.post(url, json = pload)
        print(x.text)


    def get_queryset(self):
        today = datetime.datetime.now().date()
        return ScannerData.objects.filter(
            created_at__gte=today)

    def post(self, request, *args, **kwargs):
        serializer = ScannerDataDataSerialisersv1(data=request.data)
        if serializer.is_valid():
            self.post_server(
                serializer.data['shell_no'], 
                serializer.data['cast_code'], 
                serializer.data['heat_code'], 
                serializer.data['shell_serial_no'], 
                serializer.data['location'])
        return self.create(request, *args, **kwargs)