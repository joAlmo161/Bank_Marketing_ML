# Bank Marketing Subscription Prediction Using Machine Learning

## 🔗 Links

| | |
|---|---|
| 🌐 **Live Demo** | https://ml-bank-marketing.streamlit.app |
| 💻 **GitHub Repository** | https://github.com/joAlmo161/Bank_Marketing_ML |

> Open the Live Demo in any browser; nothing needs to be installed.
> If the app is asleep, click **"Yes, get this app back up!"** and wait about a minute.

## 💡 Project Idea

A Portuguese bank called its clients to sell **term deposits**, but only **11.7 %** of them subscribed, so most calls were wasted.
This project builds a machine learning model that predicts, **before a call is made**, whether a client is likely to subscribe.
The bank can then **call the most promising clients first** and save time and effort.

## ⚙️ How It Works

1. **Data:** UCI Bank Marketing dataset (45,211 clients, 16 features).
2. **No data leakage:** the call `duration` is excluded because it is only known *after* the call.
3. **Models compared:** Dummy baseline, Logistic Regression, Decision Tree, KNN, Random Forest and Gradient Boosting.
4. **Final model:** tuned **Gradient Boosting**, deployed in a Streamlit web app.

## 📊 Results (test set)

| Metric | Value |
|---|---|
| ROC-AUC | **0.806** |
| Recall (subscribers found) | 0.540 |
| Precision | 0.467 |
| F1-score | 0.500 |

Calling the top 20 % of clients ranked by the model reaches **62.7 %** of all subscribers.

## 🖥️ How to Use the App

1. Enter the client's information (age, job, balance, loans, contact details, previous campaign).
2. Click **🔮 Predict subscription**.
3. See the result (**YES / NO**) and the probability of subscribing.

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
