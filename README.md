# Employee Attrition Predictor 🚀

Predict whether an employee is likely to leave the company — built with Python, Scikit-learn, and Streamlit.

**Live Demo:** *(Deploy using steps below — takes ~5 minutes)*

---

## Project Structure

```
├── app.py              ← Streamlit web app (main entry point)
├── train_model.py      ← ML training pipeline
├── best_model.pkl      ← Trained Logistic Regression model
├── scaler.pkl          ← StandardScaler
├── feature_names.pkl   ← Feature column names
├── model_results.json  ← Model evaluation metrics
├── HR_Employee_Attrition.csv  ← IBM HR Dataset
├── requirements.txt    ← Python dependencies
├── render.yaml         ← Render deployment config
└── README.md
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| **Logistic Regression** | 86.4% | 0.64 | 0.34 | **0.44** | **0.81** |
| Random Forest | 83.3% | 0.47 | 0.32 | 0.38 | 0.77 |
| Gradient Boosting | 85.0% | 0.59 | 0.21 | 0.31 | 0.79 |

> Best model: **Logistic Regression** (selected by F1 score for imbalanced dataset)

---

## Quick Start (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy to Render in 5 Minutes (Free) 🏆

### Step 1 — Create GitHub Repo

1. Go to [github.com](https://github.com) → Sign in → Click **"+"** → **"New repository"**
2. Name it: `attrition-predictor` (public) → Click **"Create repository"**
3. You'll see an empty repo with a URL like: `https://github.com/YOUR_USERNAME/attrition-predictor`

### Step 2 — Push Code to GitHub

Run these commands on your computer (in the folder where you extracted the files):

```bash
cd attrition_deploy
git init
git add .
git commit -m "Employee Attrition Predictor"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/attrition-predictor.git
git push -u origin main
```

Refresh your GitHub repo — all files should be there!

### Step 3 — Deploy on Render

1. Go to [render.com](https://render.com) → Click **"Get Started"** → Sign up with GitHub (free)
2. Once logged in, click **"New +"** → **"Web Service"**
3. Under **"Connect a repository"**, find and select `attrition-predictor`
4. Fill in the settings:

   | Field | Value |
   |-------|-------|
   | **Name** | `attrition-predictor` |
   | **Region** | Oregon (or nearest) |
   | **Branch** | `main` |
   | **Root Directory** | *(leave blank)* |
   | **Runtime** | `Python` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
   | **Plan** | `Free` |

5. Click **"Create Web Service"**
6. Wait 2-3 minutes → You'll see a green "Live" badge
7. Your app is live at: `https://attrition-predictor.onrender.com` 🎉

---

## OR — Deploy to Streamlit Cloud (Even Faster!)

1. Push code to a **public** GitHub repo (same as above)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"Deploy a public app"**
4. Paste: `https://github.com/YOUR_USERNAME/attrition-predictor`
5. Done! You get a URL like: `https://YOURUSERNAME-attrition-predictor.streamlit.app`

---

## What's in the App?

- **🔮 Predict** — Enter employee details → get instant attrition risk prediction with probability score
- **📊 Dashboard** — Interactive EDA: attrition by department, role, income, overtime, tenure
- **📋 Risk Factors** — Auto-identifies why the employee is at risk

---

## Resume Write-up (copy this!)

```
Employee Attrition Prediction | Live Demo: https://attrition-predictor.onrender.com
GitHub: https://github.com/YOUR_USERNAME/attrition-predictor

- Built end-to-end ML pipeline predicting employee turnover (IBM HR Dataset, 1,470 employees)
- Compared Logistic Regression, Random Forest & Gradient Boosting; Logistic Regression 
  deployed (best F1 score on imbalanced data)
- Results: 86.4% accuracy, 0.81 AUC-ROC, interactive Streamlit web app for real-time predictions
- Tech Stack: Python, Scikit-learn, Streamlit, Pandas, Matplotlib, Seaborn
```
