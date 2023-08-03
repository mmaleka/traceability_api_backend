from rest_framework import generics, mixins
from imagedetection.models import Serial_Detection
from .serializers import Serial_DetectionSerializers
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import cv2
import numpy as np
import time
import os
import re

from .base import Serial_DetectionAPIView
from .sort_bb import SortBBAlgorithm
# Create detection object
serial = Serial_DetectionAPIView()
# Create sort object
sort = SortBBAlgorithm()


class Serial_DetectionAPIView(mixins.CreateModelMixin, generics.ListAPIView):

    lookup_field = 'id'
    serializer_class = Serial_DetectionSerializers


    def computeLogoFromMemoryFILE(self, logo):
        return cv2.imdecode(np.fromstring(logo.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    def get_queryset(self):
        return Serial_Detection.objects.all()


    def post(self, request, *args, **kwargs):
        start_time2 = time.time()
        serial_number=''
        json_object = {'success': False}
        json_object['serial_number']="---"
        json_object['shell_no'] = "---"
        json_object['batch']="------"
        json_object['inference_time']=5

        if request.FILES.get("media", None) is not None:
            print("yes getting the file")
            counter = 0
            image = self.computeLogoFromMemoryFILE(request.FILES.get('media'))
            (bounding_boxes_list, scores_list, ori_image) = serial.detection(image)

            if len(bounding_boxes_list) >= 7:
                (bb_digits, bb_alphabets) = sort.sortBB(image, bounding_boxes_list)
                prediction_label_digits = serial.detect_digits(bb_digits, image, ori_image)
                prediction_label_alphabets="------"
                # if len(bb_alphabets) == 6:
                #     prediction_label_alphabets = serial.detect_alphabets(bb_alphabets, image, ori_image)
                #     print("alphabets process complete in: ", time.time()-start_time5) 


                # json_object['success'] = True
                # print("shellNo: ", prediction_label_digits, " - Batch: ", prediction_label_alphabets)
                # serial_number=str(prediction_label_digits)+'-'+prediction_label_alphabets
                
                
                # print("process complete in: ", time.time()-start_time2) 

                # json_object['serial_number'] = serial_number
                json_object['shell_no'] = prediction_label_digits
                # json_object['batch'] = prediction_label_alphabets

            else:
                print("no detections...pass")

        else:
            print("not getting the file")

        
        json_object['inference_time'] = round(time.time()-start_time2, 2)       
        print("json_object: ", json_object)   

        return JsonResponse(json_object)



