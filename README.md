### Short description 


This project is a web application designed to let users interact with various image processing techniques using a user-friendly interface. Developed with Python, Streamlit and OpenCV, it offers functionalities such as blurring, brightness adjustment, enhancing image details, and object detection.<br>

Upon uploading an image, users can effortlessly adjust parameters like blur intensity and brightness using intuitive sliders. They also have the option to enhance image details and detect objects within the uploaded image with just a click of a checkbox.<br>

Under the hood, the application utilizes the power of OpenCV. Techniques like Gaussian blur, brightness adjustment, and detail enhancement are seamlessly applied to the uploaded image based on user preferences. Object detection capabilities are integrated using a pre-trained YOLOv3 model, which identifies objects and draws bounding boxes around them. Additionally, there's an option to count the vehicles detected in the scene.<br><br>
<br>





## Requirments and actions 

Clone This Repo and open the cloned directory in an IDE  <br>
Run  ***pip install streamlit opencv-python-headless pillow***  command in terminal <br><br>
Make a file named yolov3.cfg in same directory and copy the content for this file from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg<br><br>
download yolov3 weights file from https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights and open this file in same directory (keep file name yolov3.weights) <br><br>
When you open this link to download the weights file content of this file will not be shown, you need to click **View raw** to get it downloaded <br><br>

**Now all set to run <br>**
<br>
use command __streamlit run  app.py__ to run 
