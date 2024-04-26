from ultralytics import YOLO
import multiprocessing
import cv2
if __name__ == '__main__':
    multiprocessing.freeze_support() 


#model=YOLO("yolov8n.yaml")
model=YOLO("yolov8n.pt")

#model.train(data="coco128.yaml", epochs=3, ) #train the model on your custom dataset (coco128.yaml)
#metrics=model.val()   # validate the model on the validation set
results=model( source=0,show=True,save=True,conf=.4)    # predict  in realtime vedio
#path=model.export(format="onnx") # export the model to onnx 

#in frist run if u want train the model  remove # in each line then second run do this code  