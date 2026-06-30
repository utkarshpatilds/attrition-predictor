"""
Streamlit Web App — Employee Attrition Predictor
Live ML deployment for resume portfolio
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Attrition Predictor",
    page_icon="📊",
    layout="centered"
)

# ── Load Model Artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ── Dataset Summary (for visualizations) ─────────────────────────────────────
# Re-load raw data for dashboard stats
RAW_COLS = ['Age', 'Attrition', 'BusinessTravel', 'Department', 'DistanceFromHome',
            'Education', 'EducationField', 'Gender', 'JobInvolvement', 'JobLevel',
            'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
            'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
            'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

# ── Preprocessing Helpers ──────────────────────────────────────────────────────
CATEGORICAL_OPTIONS = {
    'BusinessTravel': ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'],
    'Department': ['Human Resources', 'Research & Development', 'Sales'],
    'EducationField': ['Human Resources', 'Life Sciences', 'Marketing',
                        'Medical', 'Other', 'Technical Degree'],
    'Gender': ['Female', 'Male'],
    'JobRole': ['Healthcare Representative', 'Human Resources', 'Laboratory Technician',
                'Manager', 'Manufacturing Director', 'Research Director',
                'Research Scientist', 'Sales Executive', 'Sales Representative'],
    'MaritalStatus': ['Divorced', 'Married', 'Single'],
    'OverTime': ['No', 'Yes'],
}

def preprocess_input(input_dict):
    """Convert user inputs to model-ready DataFrame matching training pipeline."""
    # Build DataFrame matching raw column order
    df = pd.DataFrame([input_dict])

    # Drop useless columns (same as training)
    df = df.drop(columns=['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'],
                 errors='ignore')

    # Encode target (not used for prediction but keeps structure)
    # Actually we don't have Attrition as input — skip

    # One-hot encode categorical columns EXACTLY like training (drop_first=True)
    categorical_cols = [c for c in CATEGORICAL_OPTIONS.keys() if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align columns with training feature names (fill missing with 0)
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Drop any extra columns not in training
    df_encoded = df_encoded[feature_names]

    # Scale numeric features
    numeric_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education',
                   'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                   'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                   'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                   'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                   'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                   'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

    numeric_cols_in_df = [c for c in numeric_cols if c in df_encoded.columns]
    df_encoded[numeric_cols_in_df] = scaler.transform(df_encoded[numeric_cols_in_df])

    return df_encoded

# ── Sidebar Navigation ────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Dataset Dashboard", "🔮 Predict Attrition"])

# ── HOME PAGE ─────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("📊 Employee Attrition Predictor")
    st.markdown("""
    **Predict whether an employee is likely to leave the company** — powered by machine learning.

    This app uses the **IBM HR Analytics Employee Attrition dataset** (1,470 employees) and was
    trained comparing three models:
    - **Logistic Regression** ← Best performer (F1: 0.44 | AUC: 0.81)
    - Random Forest
    - Gradient Boosting

    ---
    ### 👨‍💻 Built With
    `Python` · `Scikit-learn` · `Streamlit` · `Pandas` · `Matplotlib`

    ---
    ### 📁 Project Files
    - `train_model.py` — full ML training pipeline
    - `best_model.pkl` — trained Logistic Regression model
    - `scaler.pkl` — StandardScaler for feature normalization
    - `app.py` — this Streamlit application
    """)
    st.info("👈 Use the sidebar to navigate between the Dashboard and Prediction tool!")

# ── DASHBOARD PAGE ────────────────────────────────────────────────────────────
elif page == "📊 Dataset Dashboard":
    st.title("📊 Dataset Dashboard")
    st.markdown("Key insights from the **IBM HR Attrition dataset** (1,470 employees)")

    # Load raw data for visualization
    @st.cache_data
    def load_raw():
        df = pd.read_csv("HR_Employee_Attrition.csv")
        df['Attrition_num'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        return df

    df_raw = load_raw()

    col1, col2, col3 = st.columns(3)
    with col1:
        total = len(df_raw)
        st.metric("Total Employees", total)
    with col2:
        left = (df_raw['Attrition'] == 'Yes').sum()
        st.metric("Left Company", left)
    with col3:
        rate = left / total * 100
        st.metric("Attrition Rate", f"{rate:.1f}%")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🧑‍💼 Demographics", "💰 Compensation", "📅 Tenure"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Attrition by Department")
            dept = df_raw.groupby('Department')['Attrition_num'].mean() * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            dept.sort_values().plot(kind='barh', ax=ax, color='#FF6B6B')
            ax.set_xlabel("Attrition Rate (%)")
            ax.set_title("Attrition by Department")
            st.pyplot(fig)
        with col_b:
            st.subheader("Attrition by Job Role")
            role = df_raw.groupby('JobRole')['Attrition_num'].mean() * 100
            fig, ax = plt.subplots(figsize=(6, 5))
            role.sort_values().plot(kind='barh', ax=ax, color='#4ECDC4')
            ax.set_xlabel("Attrition Rate (%)")
            ax.set_title("Attrition by Job Role")
            st.pyplot(fig)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Monthly Income Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            for label, color in [('Stayed', '#2ECC71'), ('Left', '#E74C3C')]:
                subset = df_raw[df_raw['Attrition'] == ('No' if label == 'Stayed' else 'Yes')]
                ax.hist(subset['MonthlyIncome'], bins=20, alpha=0.6, label=label, color=color)
            ax.set_xlabel("Monthly Income ($)")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)
        with col_b:
            st.subheader("Attrition by Overtime")
            ot = df_raw.groupby('OverTime')['Attrition_num'].mean() * 100
            fig, ax = plt.subplots(figsize=(5, 4))
            ot.plot(kind='bar', ax=ax, color=['#3498DB', '#E74C3C'])
            ax.set_ylabel("Attrition Rate (%)")
            ax.set_title("Overtime Impact")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            st.pyplot(fig)

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Attrition by Years at Company")
            yrs = df_raw[df_raw['Attrition'] == 'Yes'].groupby('YearsAtCompany').size()
            fig, ax = plt.subplots(figsize=(7, 4))
            yrs.plot(kind='bar', ax=ax, color='#9B59B6', alpha=0.8)
            ax.set_xlabel("Years at Company")
            ax.set_ylabel("Employees Who Left")
            ax.set_title("Who Leaves — by Tenure")
            st.pyplot(fig)
        with col_b:
            st.subheader("Attrition by Work-Life Balance")
            wlb = df_raw.groupby('WorkLifeBalance')['Attrition_num'].mean() * 100
            fig, ax = plt.subplots(figsize=(5, 4))
            wlb.plot(kind='bar', ax=ax, color='#F39C12')
            ax.set_xlabel("Work-Life Balance Rating (1=Bad, 4=Best)")
            ax.set_ylabel("Attrition Rate (%)")
            ax.set_title("Work-Life Balance Impact")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            st.pyplot(fig)

    st.markdown("---")
    st.markdown("**Key Insight:** Employees who work overtime, earn less, and have shorter tenures are most likely to leave.")

# ── PREDICTION PAGE ───────────────────────────────────────────────────────────
elif page == "🔮 Predict Attrition":
    st.title("🔮 Predict Employee Attrition")
    st.markdown("Fill in the employee details below to predict if they're likely to leave.")

    with st.form("prediction_form"):
        st.subheader("👤 Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 65, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            distance = st.slider("Distance From Home (miles)", 1, 30, 5)
        with col2:
            education = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                       format_func=lambda x: {1: "Below College", 2: "College",
                                                               3: "Bachelor", 4: "Master", 5: "Doctor"}[x])
            ed_field = st.selectbox("Education Field", CATEGORICAL_OPTIONS['EducationField'])
            dept = st.selectbox("Department", CATEGORICAL_OPTIONS['Department'])
            job_role = st.selectbox("Job Role", CATEGORICAL_OPTIONS['JobRole'])

        st.subheader("💼 Job Details")
        col1, col2 = st.columns(2)
        with col1:
            job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
            job_involvement = st.slider("Job Involvement (1=Low, 4=High)", 1, 4, 3)
            job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
            env_satisfaction = st.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, 3)
        with col2:
            overtime = st.selectbox("Overtime?", ["No", "Yes"])
            travel = st.selectbox("Business Travel", CATEGORICAL_OPTIONS['BusinessTravel'])
            work_life = st.selectbox("Work-Life Balance (1=Bad, 4=Best)", [1, 2, 3, 4],
                                      format_func=lambda x: {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}[x])
            rel_satisfaction = st.slider("Relationship Satisfaction (1=Low, 4=High)", 1, 4, 3)

        st.subheader("💰 Compensation & Growth")
        col1, col2 = st.columns(2)
        with col1:
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=500)
            daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, value=800, step=50)
            hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=100, value=65, step=5)
            monthly_rate = st.number_input("Monthly Rate ($)", min_value=1000, max_value=30000, value=15000, step=500)
        with col2:
            salary_hike = st.slider("Last Salary Hike (%)", 1, 25, 12)
            stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])
            perf_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])

        st.subheader("📅 Tenure & Experience")
        col1, col2 = st.columns(2)
        with col1:
            total_working_yrs = st.slider("Total Working Years", 0, 40, 8)
            num_companies = st.slider("Number of Companies Worked", 0, 10, 3)
            years_at_company = st.slider("Years at Company", 0, 40, 5)
        with col2:
            years_in_role = st.slider("Years in Current Role", 0, 20, 3)
            years_since_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
            years_with_manager = st.slider("Years With Current Manager", 0, 20, 3)

        training_times = st.slider("Training Times Last Year", 0, 6, 2)

        submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

    if submitted:
        # Build input dict matching the exact raw column names
        input_data = {
            'Age': age,
            'BusinessTravel': travel,
            'DailyRate': daily_rate,
            'Department': dept,
            'DistanceFromHome': distance,
            'Education': education,
            'EducationField': ed_field,
            'Gender': gender,
            'HourlyRate': hourly_rate,
            'JobInvolvement': job_involvement,
            'JobLevel': job_level,
            'JobRole': job_role,
            'JobSatisfaction': job_satisfaction,
            'MaritalStatus': marital,
            'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_rate,
            'NumCompaniesWorked': num_companies,
            'OverTime': overtime,
            'PercentSalaryHike': salary_hike,
            'PerformanceRating': perf_rating,
            'RelationshipSatisfaction': rel_satisfaction,
            'StockOptionLevel': stock_option,
            'TotalWorkingYears': total_working_yrs,
            'TrainingTimesLastYear': training_times,
            'WorkLifeBalance': work_life,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_role,
            'YearsSinceLastPromotion': years_since_promo,
            'YearsWithCurrManager': years_with_manager,
        }

        try:
            X = preprocess_input(input_data)
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]

            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            col_risk, col_prob = st.columns([1, 1])

            with col_risk:
                if prediction == 1:
                    st.error("⚠️ HIGH ATTRITION RISK — Employee is likely to LEAVE")
                    st.markdown("### 🚪 Will Likely Leave")
                else:
                    st.success("✅ LOW ATTRITION RISK — Employee is likely to STAY")
                    st.markdown("### 🏢 Will Likely Stay")

            with col_prob:
                st.markdown("**Probability Breakdown:**")
                st.write(f"- Stay (No Attrition): **{probability[0]*100:.1f}%**")
                st.write(f"- Leave (Attrition):   **{probability[1]*100:.1f}%**")

                # Progress bar
                st.progress(probability[1], text=f"Leaving Risk: {probability[1]*100:.1f}%")

            # Risk factors explanation
            st.markdown("---")
            st.subheader("📋 Key Risk Factors to Watch")
            risk_factors = []
            if overtime == "Yes":
                risk_factors.append("🔴 Works overtime — top attrition driver")
            if monthly_income < 4000:
                risk_factors.append("🔴 Low monthly income — high turnover risk")
            if years_at_company < 2:
                risk_factors.append("🟠 New employee — highest exit rate in first year")
            if work_life == 1:
                risk_factors.append("🔴 Poor work-life balance — flight risk")
            if job_involvement < 3:
                risk_factors.append("🟠 Low job involvement — disengagement sign")
            if distance > 20:
                risk_factors.append("🟠 Long commute — attrition factor")
            if stock_option == 0:
                risk_factors.append("🟠 No stock options — missing retention incentive")
            if years_since_promo > 5:
                risk_factors.append("🟠 Long time without promotion — demotivating")

            if risk_factors:
                for f in risk_factors:
                    st.markdown(f)
            else:
                st.markdown("✅ No major risk factors detected.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Try adjusting the input values.")
