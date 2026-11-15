Painting Age Predictor Using Deep Learning
Project Overview
The Painting Age Predictor is a deep learning-based application that predicts the approximate age of a painting from an uploaded image. It uses the MobileNetV2 pre-trained model and TensorFlow to extract image features and perform age prediction.
Objectives
- To develop a deep learning model for estimating the age of paintings.
- To use CNN-based feature extraction.
- To apply transfer learning using MobileNetV2.
- To provide an easy-to-use interface for image upload and prediction.
Technologies Used
- Python
- TensorFlow and Keras
- MobileNetV2
- NumPy
- Pandas
- Pillow
- Scikit-learn
- Matplotlib
- Gradio
Dataset
The dataset was collected from The Metropolitan Museum of Art Open Access Collection. It contains 1,196 valid painting images with approximate creation-year labels.
Model
MobileNetV2 is used as a feature extractor, followed by a regression layer to predict the approximate painting age.
How to Run the Project
1. Clone the repository.
2. Install the required libraries:
pip install -r requirements.txt

3. Run the application:

python app.py

4. Upload a painting image to view the estimated age.
Output
The application displays the estimated age of the uploaded painting in years.
Note
The prediction is an approximate estimate and may differ from the actual historical age of a painting.
