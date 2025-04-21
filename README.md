🌾 Rice Leaf Disease Classifier

Welcome to the Rice Leaf Disease Classifier — an AI-powered web application built with Streamlit and TensorFlow. This tool allows users, especially farmers and researchers, to upload images of rice leaves and instantly receive a prediction about the type of disease affecting the crop.

This application contributes to early detection and management of common rice diseases, helping improve crop yield and agricultural efficiency.

🚀 Live Demo

To run the app locally:

streamlit run appp.py

🧐 How It Works

Upload an image of a rice leaf (JPG, JPEG, PNG)

The app preprocesses the image and feeds it into a fine-tuned deep learning model

You receive:

Disease prediction

Confidence score

Probability breakdown for all possible classes

🧬 Model Information

The classifier is trained on high-quality rice leaf datasets and is capable of predicting the following diseases:

Bacterial Leaf Blight

Brown Spot

Leaf Smut

Model Input Size: 128x128 pixels (normalized RGB)

📁 Project Structure

rice-leaf-disease-classifier/
├── appp.py                  # Streamlit web application
├── rice_leaf_model.h5       # Pre-trained Keras model
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignored files
├── README.md                # Project documentation
└── assets/
    └── sample_leaf.jpg      # Optional sample image

📊 Results & Interface

After uploading a leaf image, you will see:

Disease name with high confidence

Confidence bar chart (percentage)

Uploaded image preview



🚜 Installation

Install dependencies using pip:

pip install -r requirements.txt

🌐 Deployment Options

This project is ready to be deployed to platforms like:

Streamlit Cloud

Render

Hugging Face Spaces

Heroku

👤 Author

Zain Arshad | mianzain450pk@gmail.com

📄 License

This project is licensed under the MIT License. Feel free to use and modify it for your own research, education, or business use.

Empowering agriculture with AI, one leaf at a time. 🤾🏼

