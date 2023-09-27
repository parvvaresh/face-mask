# Face Mask Detection using TensorFlow and OpenCV

This is a Python script for real-time face mask detection using a pre-trained MobileNetV2 model with TensorFlow and OpenCV. It can process live video from a webcam and classify faces as wearing a mask or not wearing a mask.

## Requirements

Before running the code, make sure you have the following dependencies installed:

- TensorFlow
- OpenCV
- imutils
- NumPy

You can install these dependencies using `pip`:

```bash
pip install tensorflow opencv-python imutils numpy
```

## Usage

1. Clone the repository or download the code files to your local machine.

2. Download the pre-trained face detection model and save it in the `face_detector` directory. You can get the model files from the OpenCV GitHub repository:
   - [deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
   - [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

3. Download the pre-trained face mask detection model (`mask_detector.model`) and place it in the same directory as the script.

4. Run the script:

```bash
python face_mask_detection.py
```

5. A window will open, showing the webcam feed with face mask predictions. Press 'q' to quit the application.

## How It Works

- The script uses the MobileNetV2 model for face mask detection. It first detects faces in each frame using the pre-trained face detection model.

- If a face is detected, it extracts the face region, pre-processes it, and passes it through the face mask detection model to classify whether the person is wearing a mask or not.

- The result is displayed on the video feed, with bounding boxes and labels.

## License

This project is licensed under the [MIT License](LICENSE).
