# Email/SMS Spam Classifier

A machine learning-based web application that classifies emails or SMS messages as spam or not spam (ham) using Natural Language Processing (NLP) and the Naive Bayes algorithm.

## 🌟 Features

- Real-time classification of email/SMS messages
- User-friendly web interface built with Streamlit
- Text preprocessing using NLTK
- Machine learning model trained on spam dataset
- Deployed on Streamlit Cloud

## 🛠️ Technologies Used

- Python 3.x
- Streamlit (Web Framework)
- NLTK (Natural Language Processing)
- Scikit-learn (Machine Learning)
- Pandas (Data Processing)
- NumPy (Numerical Operations)

## 📁 Project Structure

```
Email-Spam-Classifier/
├── app.py                 # Main Streamlit application file
├── model.pkl             # Trained machine learning model
├── vectorizer.pkl        # TF-IDF vectorizer
├── spam.csv             # Dataset used for training
├── sms-spam-detection.ipynb  # Jupyter notebook with model development
├── requirements.txt      # Python dependencies
├── Procfile             # Deployment configuration for cloud platforms
├── setup.sh             # Setup script for deployment
└── nltk.txt             # NLTK requirements
```

## 🚀 How It Works

1. **Text Preprocessing**:
   - Converts text to lowercase
   - Tokenization
   - Removes special characters and numbers
   - Removes stopwords and punctuation
   - Applies Porter Stemming

2. **Feature Extraction**:
   - Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
   - Transforms text data into numerical features

3. **Classification**:
   - Employs a trained Naive Bayes classifier
   - Predicts whether the input message is spam or not

## 💻 Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Email-Spam-Classifier.git
   cd Email-Spam-Classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`

## 📊 Model Training

The model was trained using:
- A dataset of labeled spam and ham messages (`spam.csv`)
- Text preprocessing techniques including tokenization and stemming
- TF-IDF vectorization for feature extraction
- Naive Bayes classification algorithm

The complete model development process can be found in `sms-spam-detection.ipynb`.

## 🌐 Deployment

The application is configured for deployment on cloud platforms with the following files:
- `Procfile`: Specifies the commands to run the app
- `setup.sh`: Contains the configuration for Streamlit
- `requirements.txt`: Lists all Python dependencies

## 🤝 Contributing

Feel free to fork this repository, create a feature branch, and submit a Pull Request.


This project is open source and available under the MIT License.

## 👥 Author

Hafiz Haris Mehmood

---
⭐️ If you find this project helpful, please give it a star! 