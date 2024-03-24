short description 


This project is a web application designed to let users interact with various image processing techniques using a user-friendly interface. Developed with Python, Streamlit and OpenCV, it offers functionalities such as blurring, brightness adjustment, enhancing image details, and object detection.

Upon uploading an image, users can effortlessly adjust parameters like blur intensity and brightness using intuitive sliders. They also have the option to enhance image details and detect objects within the uploaded image with just a click of a checkbox.

Under the hood, the application utilizes the power of OpenCV. Techniques like Gaussian blur, brightness adjustment, and detail enhancement are seamlessly applied to the uploaded image based on user preferences. Object detection capabilities are integrated using a pre-trained YOLOv3 model, which identifies objects and draws bounding boxes around them. Additionally, there's an option to count the vehicles detected in the scene.





### How to run this 
Clone This Repo 
Run  ***pip install streamlit opencv-python-headless pillow***  command in terminal 
Make a file named yolov3.cfg in directory and copy the the content from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
download yolov3 weights file from https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights and open this file in same directory 
When you open this link to download the weights file the content of this file will not be shown, you need to click **View raw** to get it downloaded 

## Now all set to run 

use command ***streamlit run  app.py*** to run 
