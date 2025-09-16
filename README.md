
# 📊 Telecom Customer Churn Detector

This project predicts whether a customer will churn (leave the service) based on their demographic and service usage details. It combines **machine learning** experimentation with a clean, user-friendly **Tkinter GUI**.

---

## 🔍 Key Work

* **Exploratory Data Analysis (EDA):**

  * Explored churn distribution across categorical and numerical features.
  * Visualized churn percentages by features such as contract type, payment method, internet service, etc.
  * Converted and cleaned missing `TotalCharges` values.

* **Modeling Journey:**

  * Tested **Decision Tree**, **Random Forest**, **Logistic Regression**, and **XGBoost**.
  * Tried different optimization strategies → Randomized Search, Grid Search, and finally **Optuna**.
  * Custom scoring function used: `f0.75` (`fbeta_score` with β=0.75) → slightly favors **precision** while still balancing **recall**.
  * Final choice: **Optuna-tuned XGBoost Classifier**, which gave the best trade-off.

---

## 🎯 Why f0.75 Instead of Accuracy?

The dataset is **imbalanced** (more customers stay than churn).

* A high **accuracy** can be misleading (e.g., predicting “No churn” for everyone gives \~73% accuracy).
* **Recall** is important (don’t miss churners), but focusing only on it increases false alarms.
* **Precision** is also important (avoid wrongly flagging loyal customers as churners).

By choosing **f0.75**, we slightly weigh precision more than recall, giving a **balanced, business-friendly metric** that reflects the real-world cost of churn prediction.

---

## 🖥 GUI

The GUI allows both **technical and non-technical users** to interact with the model easily.

### Features:

* Manual form input (dropdowns + sliders for numerical values).
* Upload `.txt` or `.csv` file in the specified format.
* Prediction with churn probability and color-coded feedback.
* Progress bar and tooltips for ease of use.
* Reset option and fullscreen support.

---

## 📂 Input File Format

When using **Upload Data**, the `.txt` or `.csv` file must follow this exact order:

```
gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,
OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,
PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
```

Example:

```
Male,0,Yes,No,12,Yes,No,Fiber optic,No,Yes,No,No,Yes,No,Month-to-month,Yes,Electronic check,70.35,845.5
```

---

## 📈 Model Performance

Final selected model: **Optuna-tuned XGBoost Classifier**

| Dataset          | Accuracy | Precision | Recall | F1 Score |
| ---------------- | -------- | --------- | ------ | -------- |
| Training         | 79.94%   | 59.52%    | 71.00% | 64.75%   |
| Cross-Validation | 80.97%   | 62.03%    | 73.08% | 67.10%   |
| Testing          | 78.04%   | 58.42%    | 67.94% | 62.82%   |

## 🛠 Tech Stack

* **Python**
* **XGBoost, Scikit-learn, Optuna** → Machine Learning
* **Tkinter** → GUI
* **Seaborn, Matplotlib, Pandas** → Data Exploration

---

## ✨ Author

Made with ❤️ by **Habib Ghulam Bheek**

---

This version is crisp, professional, and has **just enough storytelling** to prove you *know why you made certain decisions*.
