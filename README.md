# DogHealth AI: Dog Skin Disease Predictor

## Project Overview

DogHealth AI is an innovative application that uses advanced AI and computer vision techniques to predict and detect skin diseases in dogs through image analysis. Our system analyzes images of dogs to identify potential health issues early, improving treatment outcomes and pet well-being.

## Live Demo

Experience DogHealth AI in action: [DogHealth AI on Streamlit](https://petponks-ai-competition-uc5xwa6ltapejqlradodcu.streamlit.app/)

## Features

- Upload images of dogs for analysis
- Predict potential skin diseases with confidence scores
- Provide descriptions of detected conditions
- User-friendly interface for both pet owners and veterinarians

## Technologies Used

- Python 3.8+
- TensorFlow 2.x
- Keras
- Streamlit
- InceptionV3 architecture
- Pillow (PIL)
- NumPy

## Installation and Local Setup

1. Clone the repository:

    ```sh
    git clone https://github.com/akhilesh1709/PetPonks-AI-competition.git
    cd PetPonks-AI-competition
    ```
2. Set up a virtual environment (optional but recommended):
   ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Download the trained model:
- Ensure `improved_dog_disease_prediction_model.h5` is in the `model_training_notebook` directory

## Running the Application

1. Start the Streamlit app:
   ```sh
   streamlit run app.py
   ```
2. Open your web browser and navigate to `http://localhost:8501`

3. Upload a dog image and click 'Predict' to get the analysis results

## Model Training

The model uses a fine-tuned InceptionV3 architecture. For details on the training process, refer to `model.ipynb`.

## Contributing

We welcome contributions to improve DogHealth AI. Please feel free to fork the repository, make changes, and submit pull requests.
