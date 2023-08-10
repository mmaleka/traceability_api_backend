import cv2
import numpy as np
import time
import os
import re
from datetime import datetime

# import tensorflow as tf
# import tensorflow_hub as hub
# # from core import utils
# # from core.config import cfg
# # from tensorflow.python.saved_model import tag_constants


from ultralytics import YOLO
import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm

from .TinyVGG import TinyVGG
from .CustomCNN import CustomCNN



class BengaliParquetDataset(Dataset):
    
    def __init__(self, image_array_list, transform=None):
        
        self.data = image_array_list
        self.trasnform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        img = Image.fromarray(self.data[idx])
        data_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        image = data_transform(img)
        
        return image


class Serial_DetectionAPIView():

    # Instance attribute
    def __init__(self):
        print("loading model")

        # Setup device-agnostic code
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO("imagedetection/api/best.pt")
        self.loaded_model_digits =  TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                        hidden_units=10, 
                        output_shape=10).to(self.device)
        
        self.loaded_model_digits.load_state_dict(torch.load('imagedetection/api/model_digits.pth', map_location=torch.device('cpu')))
        self.loaded_model_digits.eval()


        self.loaded_model_alphabets =  CustomCNN(input_shape=3, # number of color channels (3 for RGB) 
                        hidden_units=10, 
                        output_shape=23).to(self.device)
        
        self.loaded_model_alphabets.load_state_dict(torch.load('imagedetection/api/model_alphabets_31.pth', map_location=torch.device('cpu')))
        self.loaded_model_alphabets.eval()

        # self.loaded_model_digits = tf.keras.models.load_model(
        #     ('imagedetection/api/my_model_digits_v3_3.h5'),
        #     custom_objects={'KerasLayer':hub.KerasLayer}
        # )

        # self.loaded_model_aphabets = tf.keras.models.load_model(
        #     ('imagedetection/api/my_model_alphabets_v3_5.h5'),
        #     custom_objects={'KerasLayer':hub.KerasLayer}
        # )


        print("done loading model")

        


    def pytorch_inference(self, X, model, BATCH_SIZE=32):
         
        test_dataset = BengaliParquetDataset(
            image_array_list=X,
        )

        data_loader_test = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        with torch.inference_mode():
            y_preds = []
            for X_in in tqdm(data_loader_test, desc="Making predictions"):
                # Send data and targets to target device
                X_in = X_in.to(self.device)
                # Do the forward pass
                y_logit = model(X_in)
                # Turn predictions from logits -> prediction probabilities -> predictions labels
                y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
                # Put predictions on CPU for evaluation
                y_preds.append(y_pred.cpu())


        return y_preds



    


    def detection(self, original_image):

        start_time1 = time.time()
        results = self.model.predict(original_image, conf=0.55)
        result = results[0]

        bounding_boxes_list = []
        ori_image = original_image.copy()
        for i in range(len(result.boxes)):
            box = result.boxes[i]
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)

            xmin=cords[0]
            ymin=cords[1]
            xmax=cords[2]
            ymax=cords[3]
            bounding_boxes_list.append((cords[0], cords[1], cords[2], cords[3]))

            # crop_path = r"C:\Users\MMaleka\Desktop\traceability_api_backend\detections\complete\test_1"
            # cropped_img = ori_image[int(ymin)-1:int(ymax)+1, int(xmin)-1:int(xmax)+1]
            # now = datetime.now()
            # current_time = now.strftime("%H:%M:%S")
            # img_name = str(i)+"_"+str(current_time).replace(":", "-")+'.jpg'
            # img_path = os.path.join(crop_path, img_name)
            # cv2.imwrite(img_path, cropped_img)

            # cv2.rectangle(
            #     original_image, 
            #     (int(xmin)-4, int(ymin)-4), 
            #     (int(xmax)+4, int(ymax)+4), 
            #     (0, 0, 255), 
            #     2
            # )

            # cv2.putText(
            #     original_image, 
            #     str(conf), 
            #     (int(xmin)+50, int(ymin) - 15), 
            #     cv2.FONT_HERSHEY_PLAIN, 
            #     2, 
            #     (0, 0, 255), 
            #     2
            # )

        # crop_path = r"C:\Users\MMaleka\Desktop\train_2"
        # now = datetime.now()
        # current_time = now.strftime("%H:%M:%S")
        # img_name = str(current_time).replace(":", "-")+'.jpg'
        # img_path = os.path.join(crop_path, img_name)
        # cv2.imwrite(img_path, original_image)

        print("box - process is complete: ", time.time()-start_time1)
        scores_list=[]
        return bounding_boxes_list, scores_list, ori_image



    
    def detect_digits(self, bb_digits, frame, original_image):

        labels = ['0','1','2','3','4','5','6','7','8','9']
        bb_digits_sorted=sorted(bb_digits, key=lambda x: x[0])
        shell_no = ''
        X = []
        for i in range(len(bb_digits_sorted)):
            xmin=bb_digits_sorted[i][0]
            ymin=bb_digits_sorted[i][1]
            xmax=xmin+bb_digits_sorted[i][2]
            ymax=ymin+bb_digits_sorted[i][3]
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            cropped_img = original_image[int(ymin)-1:int(ymax)+1, int(xmin)-1:int(xmax)+1]
            resized_img_test = cv2.resize(cropped_img,(64,64))
            X.append(resized_img_test)

        X = np.array(X)
        y_preds = self.pytorch_inference(X, self.loaded_model_digits)
        for index in y_preds[0].tolist():
            prediction_label = labels[index].upper()
            shell_no += prediction_label

        

                
                # resized_img_test = cv2.resize(cropped_img,(50,50))
                # resized_img_test_scale = resized_img_test / 255
                # X3.append(resized_img_test_scale)
                # x = np.expand_dims(resized_img_test_scale, axis=0)
                # prediction = self.loaded_model_digits.predict(x)
                # y_predicted_labels = [np.argmax(i) for i in prediction]
                # y_predicted_labels = np.array(y_predicted_labels)
                # prediction_label = labels[y_predicted_labels[0]].upper()

                # crop_path = r"C:\Users\MMaleka\Desktop\traceability_api_backend\detections\complete\digits\{}".format(prediction_label)
                # # Check if the directory exists
                # if not os.path.exists(crop_path):
                #     # If it doesn't exist, create it
                #     os.makedirs(crop_path)

                
                # img_name = str(prediction_label)+"_"+str(current_time).replace(":", "-")+'.jpg'
                # img_path = os.path.join(crop_path, img_name)
                # cv2.imwrite(img_path, cropped_img)

                # cv2.rectangle(
                #     original_image, 
                #     (int(xmin)-4, int(ymin)-4), 
                #     (int(xmax)+4, int(ymax)+4), 
                #     (0, 0, 255), 
                #     2
                #     )
                # cv2.putText(
                #     frame, 
                #     prediction_label, 
                #     (int(xmin), int(ymin) - 15), 
                #     cv2.FONT_HERSHEY_PLAIN, 
                #     2, 
                #     (255, 0, 0), 
                #     2
                #     )

                # save image to retrain the model.
                # This is the complete image
                # crop_path = r"C:\Users\MMaleka\Desktop\traceability_api_backend\detections\complete"
                # img_name = str(i)+"_"+str(current_time).replace(":", "-")+'.jpg'
                # img_path = os.path.join(crop_path, img_name)
                # cv2.imwrite(img_path, cropped_img)

                

            #     shell_no = shell_no+prediction_label

            # except Exception as e:
            #     shell_no="---"
            #     print("error detecting digits: ", e)


        # start_time2 = time.time()
        # try:
        #     X3 = np.array(X3)
        #     print("X3.shape: ", X3.shape)
        #     predictions=self.loaded_model_digits.predict(X3)
        #     y_predicted_labels = [np.argmax(i) for i in predictions]
        #     print(y_predicted_labels)
        #     for index in y_predicted_labels:
        #         prediction_label = labels[index].upper()
        #         shell_no = shell_no+prediction_label
        # except Exception as e:
        #         shell_no="---"
        #         print("error detecting digits: ", e)
        # print("digits1 process complete in: ", time.time()-start_time2)

        
        return shell_no
        
    


    def detect_alphabets(self, bb_alphabets, frame, original_image):

        labels = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','R','S','T','U','V','W','X','Y','Z']
        bb_alphabets_sorted=sorted(bb_alphabets, key=lambda x: x[0])
        batch = ''
        X = []
        for i in range(len(bb_alphabets_sorted)):
            xmin=bb_alphabets_sorted[i][0]
            ymin=bb_alphabets_sorted[i][1]
            xmax=xmin+bb_alphabets_sorted[i][2]
            ymax=ymin+bb_alphabets_sorted[i][3]
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            cropped_img = original_image[int(ymin)-1:int(ymax)+1, int(xmin)-1:int(xmax)+1]
            resized_img_test = cv2.resize(cropped_img,(64,64))
            X.append(resized_img_test)
        

        X = np.array(X)
        y_preds = self.pytorch_inference(X, self.loaded_model_alphabets)
        print(y_preds[0].tolist())
        for index in y_preds[0].tolist():
            prediction_label = labels[index].upper()
            batch += prediction_label


            # resized_img_test = cv2.resize(cropped_img,(50,50))
            # resized_img_test_scale = resized_img_test / 255
            # x = np.expand_dims(resized_img_test_scale, axis=0)
            # X2.append(resized_img_test_scale)

            # prediction = self.loaded_model_aphabets.predict(x)
            # # prediction_1 = self.loaded_model_aphabets_v1.predict(x)
            # # prediction_2 = self.loaded_model_aphabets_v2.predict(x)
            
            # y_predicted_labels = [np.argmax(i) for i in prediction]
            # # y_predicted_labels_1 = [np.argmax(i) for i in prediction]
            # # y_predicted_labels_2 = [np.argmax(i) for i in prediction]
            # y_predicted_labels = np.array(y_predicted_labels)
            # # y_predicted_labels_1 = np.array(y_predicted_labels_1)
            # # y_predicted_labels_2 = np.array(y_predicted_labels_2)

            # # softmax = tf.nn.softmax(prediction)
            # # print("y_predicted_labels: ", y_predicted_labels[0]," softmax: ", softmax[y_predicted_labels[0]])

            # prediction_label = labels[y_predicted_labels[0]].upper()
            # prediction_label_1 = labels[y_predicted_labels_1[0]].upper()
            # prediction_label_2 = labels[y_predicted_labels_2[0]].upper()

            # # save image to retrain the model.
            # crop_path = r"C:\Users\MMaleka\Desktop\traceability_api_backend\detections\complete\alphabets\{}".format(prediction_label)
            # # Check if the directory exists
            # if not os.path.exists(crop_path):
            #     # If it doesn't exist, create it
            #     os.makedirs(crop_path)

            # img_name = str(prediction_label)+"_"+str(current_time).replace(":", "-")+'.jpg'
            # img_path = os.path.join(crop_path, img_name)
            # cv2.imwrite(img_path, cropped_img)

            # cv2.rectangle(
            #     original_image, 
            #     (int(xmin)-2, int(ymin)-2), 
            #     (int(xmax)+2, int(ymax)+2), 
            #     (0, 0, 255), 
            #     2
            #     )
            # cv2.putText(
            #     frame, 
            #     prediction_label, 
            #     (int(xmin), int(ymax)+40), 
            #     cv2.FONT_HERSHEY_PLAIN, 
            #     2, 
            #     (255, 0, 0), 
            #     2
            #     )

            # crop_path = r"C:\Users\MMaleka\Desktop\traceability_api_backend\detections\complete\test_1"
            # # cropped_img = ori_image[int(ymin)-1:int(ymax)+1, int(xmin)-1:int(xmax)+1]
            # now = datetime.now()
            # current_time = now.strftime("%H:%M:%S")
            # img_name = str(current_time).replace(":", "-")+'.jpg'
            # img_path = os.path.join(crop_path, img_name)
            # cv2.imwrite(img_path, frame)

            # batch = batch+prediction_label


        # start_time2 = time.time()
        # try:
        #     X2 = np.array(X2)
        #     print("X2.shape: ", X2.shape)
        #     predictions=self.loaded_model_aphabets.predict(X2)
        #     y_predicted_labels = [np.argmax(i) for i in predictions]
        #     print(y_predicted_labels)
        #     for index in y_predicted_labels:
        #         prediction_label = labels[index].upper()
        #         batch = batch+prediction_label
        # except Exception as e:
        #         batch="------"
        #         print("error detecting letters: ", e)
        # print("alphabets1 process complete in: ", time.time()-start_time2)


        return batch
    



    def speak(self, name):
        print("My name is {}".format(name))




# serial = Serial_DetectionAPIView()
# img = cv2.imread('2008.jpg')
# (bounding_boxes_list, scores_list) = serial.detection(img)












