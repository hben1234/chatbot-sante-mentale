#  Mental Health Support Assistant (NLP & Sentiment Analysis)

An AI-driven platform designed to provide supportive orientation and sentiment detection for individuals seeking mental health resources. This project combines a responsive web interface with a multi-model machine learning backend.

##  Key Features
* **Intelligent Chatbot:** Real-time dialogue system using NLTK-based pattern matching for common mental health inquiries.
* **Sentiment Analysis Pipeline:** Automated text preprocessing (tokenization, stop-word removal, contraction expansion) followed by sentiment prediction.
* **Machine Learning Benchmarking:** Evaluated 4 distinct algorithms (SVM, Naive Bayes, Decision Tree, Logistic Regression) to optimize intent classification accuracy.
* **Full-Stack Integration:** A Flask-based REST API serving a complete 15-page responsive website.

## Technical Stack
* **Language:** Python
* **NLP:** NLTK, Spacy, Contractions
* **Machine Learning:** Scikit-learn (SVM, TF-IDF Vectorization)
* **Backend:** Flask, Flask-CORS
* **Frontend:** HTML5, CSS3, JavaScript, Bootstrap

## Model Performance
The primary SVM model was trained on the `mental_health.csv` dataset, utilizing a TF-IDF pipeline to handle text vectorization.

## Project Structure
* `/app.py`: Main Flask application and chatbot logic.
* `/data`: Contains the mental health training dataset.
* `/models`: Serialized `.pkl` model artifacts.
* `/scripts`: Python scripts for model training and benchmarking.
* `/templates & /static`: UI components and assets.

## 📸 Demo & Performance
|  Model Accuracy |
|  :---: |
| ![Accuracy](assets/Screenshots/model_accuracy) |
| *SVM Model Validation Metrics* |

---


## 📊 Model Benchmarks
| Algorithm | Accuracy | Role |
| :--- | :---: | :--- |
| **SVM** | **92,64%** | Production Model |
| **Logistic Regression** | 92,16% | Benchmark |
| **Naive Bayes** | 84% | Benchmark |
| **Decision Tree** | 83% | Benchmark |

---

## Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the application: `python app.py`.
