# Student Score Prediction using Linear Regression

A machine learning project that predicts students' exam scores based on various performance factors.

## 📊 Dataset
- **Source**: [Student Performance Factors (Kaggle)](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
- **Features**: Hours studied, attendance, sleep hours, previous scores, tutoring sessions, physical activity
- **Target**: Exam scores

## 🛠️ Technologies Used
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

## 📁 Project Structure
```
├── main.py                           # Main script
├── Student_Performance.csv           # Dataset (download separately)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/ML-Task1-Student-Score-Prediction.git
cd ML-Task1-Student-Score-Prediction
```

2. **Create a virtual environment:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
- Download from [Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
- Save as `Student_Performance.csv` in the project folder

5. **Run the script:**
```bash
python main.py
```

## 📈 Features
- Data cleaning and preprocessing
- Exploratory data analysis with visualizations
- Linear regression model training
- Model performance evaluation (R², MAE, RMSE)
- Feature importance analysis
- **Bonus**: Polynomial regression comparison

## 📊 Results
The model achieves strong performance in predicting exam scores, with visualizations showing:
- Actual vs Predicted scores
- Residual plots
- Feature importance rankings

## 📝 License
This project is open source and available under the MIT License.

## 👤 Author
Your Name - [GitHub Profile](https://github.com/YOUR_USERNAME)
