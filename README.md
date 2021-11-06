# Road-Defects-Detection

Computer vision-based system for real-time detection and localization of road surface defects such as potholes and cracks, is proposed to overcome the limitations and inefficiency of human-based visual onsite inspections

Detection is done using TensorFlow Lite model, installed on Raspberry pi that is integrated with camera and GPS modules. System connected to a cloud database where the image with detected defect is uploaded with its precise location and type.



### Proposed Architectures

#### TensorFlow Lite:
We use EfficientDet-Lite[0-4], which are a family of mobile/IoT-friendly object detection models derived from the EfficientDet architecture, as a base model for transfer learning.
The dataset is split up into training and testing sets where 90% of the data is used as training data and 10% used as validation and testing data.
Implementation of training phase was with (300 epochs, batch size 32, and image size 320×320 pixels) for the model.

#### YOLOv3:
A customized YOLOv3 architecture with Darknet-53 for our labeled dataset and custom model.
In YOLO architecture, we use Convolutional Neural Network (CNN) for the road defects detection as the positive marked regions represent road defects.
The dataset is split up into training and testing sets where 85% for training data and 15% for testing data.
We’ve trained different models on Google Collaboratory with the mentioned architecture and hyperparameters.
Implementation of training phase was with (8000 epochs, batch size 64, and image size 418×418 pixels) for the model.
