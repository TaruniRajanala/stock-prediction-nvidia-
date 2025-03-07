# 📈  Stock Prediction (Nvidia)

### 🔍 Overview
This project aims to predict NVIDIA's stock prices using machine learning models. We leverage **historical stock data**, perform **exploratory data analysis (EDA)**, and implement **time series forecasting** techniques to make future price predictions.

### 📂 Project Structure
```
📁 Nvidia-Stock-Prediction  
│-- 📄 README.md  
│-- 📄 stock-prediction.ipynb  # Jupyter Notebook with EDA & Model  
│-- 📄 finance_news.ipynb  # Sentiment Analysis on Finance News  
│-- 📄 reddit_sentiment_analysis.ipynb  # Reddit-based Sentiment Analysis  
│-- 📂 data  
│   ├── avg_news_sentiment.csv  # Processed Sentiment Data  
│   ├── nvda_news.csv  # Raw Finance News Data  
│   ├── train_reddit_df_sentiment.csv  # Labeled Reddit Sentiment Data  
│-- 📂 models  
│   ├── trained_model.pkl  # Saved ML model  
│-- 📂 images  
│   ├── stock_trend_plot.png  # Visualization of stock trends  
```

### 🛠️ Tech Stack
- Python 🐍  
- Pandas & NumPy (Data Processing)  
- Matplotlib & Seaborn (Data Visualization)  
- Scikit-learn (ML Models)  
- TensorFlow/Keras (LSTM for Time Series)  
- Natural Language Toolkit (NLTK) (Sentiment Analysis)  

### 📊 Data Sources
- Historical stock data from [Yahoo Finance](https://finance.yahoo.com/) using the `yfinance` API.  
- News and Reddit sentiment data processed for predictive modeling.  

### 📉 Methodology
1. **Data Collection** – Fetch historical stock prices & financial news.  
2. **EDA** – Identify trends, moving averages, and seasonality.  
3. **Sentiment Analysis** – Analyze financial news & Reddit discussions.  
4. **Feature Engineering** – Create lag features, rolling averages, sentiment scores.  
5. **Modeling** – Train ML models like Linear Regression, LSTM, and ARIMA.  
6. **Evaluation** – Measure RMSE, MSE for accuracy.  

### 📌 Results & Findings
- The LSTM model performed best with an **RMSE of X.XX**.  
- Sentiment Analysis showed a correlation between stock price movements and financial news sentiment.  

### 🚀 How to Run?
#### 1️⃣ Install Dependencies
```bash
pip install pandas numpy matplotlib scikit-learn yfinance tensorflow nltk
```
#### 2️⃣ Run the Jupyter Notebook
```bash
jupyter notebook
```
Open `stock-prediction.ipynb` and execute the cells.  

#### 3️⃣ Run Python Script (if available)
```bash
python stock_prediction.py
```

### 📌 Future Improvements
- Fine-tune hyperparameters for better accuracy  
- Incorporate **real-time** news sentiment analysis  
- Deploy model as a **Flask API**  

🔗 **Check Out the Full Project** [Insert GitHub Repo Link]  

📩 **Have Suggestions? Open an Issue or Fork the Repo!**  
