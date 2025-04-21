🌾 Rice Leaf Disease Classifier
A deep learning-based web application to classify rice leaf diseases using TensorFlow and Streamlit. The model identifies three types of diseases: Bacterial Leaf Blight, Brown Spot, and Leaf Smut.

Streamlit
License

📌 Features
Deep Learning Model: Uses transfer learning with MobileNetV2 for high accuracy.

Web Interface: Streamlit-based UI for easy image upload and prediction.

Data Augmentation: Enhances training data with rotations, flips, and zooms.

Confidence Scores: Displays class probabilities alongside predictions.

Evaluation Metrics: Includes accuracy plots and confusion matrix.

🛠️ Installation
Prerequisites
Python 3.8+

pip

Steps
Clone the Repository

bash
git clone https://github.com/Mian-Zain/rice-leaf-disease-classifier.git
cd rice-leaf-disease-classifier
Install Dependencies

bash
pip install -r requirements.txt
Download the Dataset

Organize your dataset into train and test folders with subdirectories for each class.

Example structure:

rice_leaf_diseases/
  train/
    Bacterial leaf blight/
    Brown spot/
    Leaf smut/
  test/
    Bacterial leaf blight/
    Brown spot/
    Leaf smut/
🚀 Usage
1. Model Training
Run the Jupyter Notebook projectt.ipynb to train the model.

The trained model will be saved as rice_leaf_model.h5.

2. Streamlit Web App
Start the app:

bash
streamlit run appp.py
Upload a rice leaf image through the interface to get predictions.

📊 Dataset & Model
Dataset: Contains 3 classes of rice leaf diseases. Adjust the paths in projectt.ipynb to match your dataset location.

Model Architecture:

python
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
model = Sequential([
    base_model,
    Flatten(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
📈 Results
Test Accuracy: ~57% (Improve this by training on a larger dataset).

Confusion Matrix:
Confusion Matrix

🤝 Contributing
Fork the repository.

Create a new branch:

bash
git checkout -b feature/your-feature
Commit changes and push to your branch.

Submit a Pull Request.

📜 License
This project is licensed under the MIT License. See LICENSE for details.

🙏 Acknowledgments
Dataset inspiration from Kaggle.

Built with TensorFlow, Streamlit, and OpenCV.

Improve Crop Health with AI! 🌱
For questions, contact [your-email@example.com].
