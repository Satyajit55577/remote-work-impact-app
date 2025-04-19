import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Set Streamlit page config
st.set_page_config(
    page_title="Remote Work Impact Analyzer",
    layout="wide"
)

# Title and subtitle
st.title("💼 Remote Work Impact Analyzer")
st.markdown("""
Welcome to the *Remote Work Impact Analyzer* — a smart app powered by Machine Learning  
to understand how remote work influences employee *mental health*, **productivity**, and **job satisfaction**.

Explore key insights and run real-time predictions using data collected from working professionals.
""")

# Load cleaned dataset
df = pd.read_csv("cleaned_remote_work_dataset_for_building_dashboard.csv")

# ------------------------
# Insight 1
# ------------------------
st.markdown("""
### 📊 Insight #1: Mental Health vs Stress Level
*People with high, medium, and low stress show almost the same levels of anxiety, burnout, and depression.*

👉 *Takeaway: In this data, stress level doesn't strongly affect mental health condition.*
""")
fig1, ax1 = plt.subplots(figsize=(4, 2.5))
pivot1 = pd.crosstab(df['Stress_Level'], df['Mental_Health_Condition'], normalize='index')
pivot1.plot(kind='bar', stacked=True, ax=ax1)
ax1.set_ylabel("Proportion", fontsize=7)
ax1.set_title("Mental Health by Stress Level", fontsize=7)
ax1.legend(title='Condition', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=5, title_fontsize=5)
plt.xticks(rotation=0, fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
st.pyplot(fig1)

# ------------------------
# Insight 2
# ------------------------
st.markdown("""
### 📈 Insight #2: Productivity Change vs Work Location
*Whether people work remotely, onsite, or hybrid, their productivity changes are very similar.*

👉 *Takeaway: Work location doesn't have much effect on productivity change.*
""")
fig2, ax2 = plt.subplots(figsize=(4, 2.5))
pivot2 = pd.crosstab(df['Work_Location'], df['Productivity_Change'], normalize='index')
pivot2.plot(kind='bar', stacked=True, ax=ax2)
ax2.set_ylabel("Proportion", fontsize=7)
ax2.set_title("Productivity by Work Location", fontsize=7)
ax2.legend(title='Productivity', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=5, title_fontsize=5)
plt.xticks(rotation=0, fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
st.pyplot(fig2)

# ------------------------
# Insight 3
# ------------------------
st.markdown("""
### 🚀 Insight #3: Satisfaction with Remote Work vs Work Location
*Remote workers are a bit more neutral. Onsite and hybrid workers are slightly more satisfied.*

👉 *Takeaway: Remote workers are less positive about remote work than others.*
""")
fig3, ax3 = plt.subplots(figsize=(4, 2.5))
pivot3 = pd.crosstab(df['Work_Location'], df['Satisfaction_with_Remote_Work'], normalize='index')
pivot3.plot(kind='bar', stacked=True, ax=ax3)
ax3.set_ylabel("Proportion", fontsize=7)
ax3.set_title("Satisfaction by Work Location", fontsize=7)
ax3.legend(title='Satisfaction', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=5, title_fontsize=5)
plt.xticks(rotation=0, fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
st.pyplot(fig3)

# ------------------------
# Model 1: Predict Mental Health Condition
# ------------------------
st.markdown("## 🧠 Model 1: Predict Mental Health Condition")
with open("model1_mental_health_bundle.pkl", "rb") as f:
    bundle1 = pickle.load(f)
model1 = bundle1["model"]
features1 = bundle1["selected_features"]
scaler1 = bundle1["scaler"]
numeric1 = ['Age', 'Years_of_Experience', 'Hours_Worked_Per_Week',
            'Number_of_Virtual_Meetings', 'Work_Life_Balance_Rating',
            'Social_Isolation_Rating', 'Company_Support_for_Remote_Work', 'Stress_Level']
controls1 = {
    'Age': (18, 65), 'Years_of_Experience': (0, 40), 'Hours_Worked_Per_Week': (20, 100),
    'Number_of_Virtual_Meetings': list(range(0, 11)),
    'Work_Life_Balance_Rating': [1, 2, 3, 4, 5], 'Stress_Level': [1, 2, 3, 4, 5],
    'Social_Isolation_Rating': [1, 2, 3, 4, 5], 'Company_Support_for_Remote_Work': [1, 2, 3, 4, 5],
}
input1 = {}
for col in features1:
    if col in controls1:
        val = controls1[col]
        input1[col] = st.selectbox(col, val) if isinstance(val, list) else st.number_input(f"{col} ({val[0]}–{val[1]})", min_value=val[0], max_value=val[1], value=val[0])
    else:
        input1[col] = st.selectbox(col, [0, 1], format_func=lambda x: "Yes" if x else "No")
if st.button("🎯 Predict Mental Health"):
    df1 = pd.DataFrame([input1])
    df1[numeric1] = scaler1.transform(df1[numeric1].to_numpy())
    st.success(f"Predicted Condition: {model1.predict(df1[features1])[0]}")

# ------------------------
# Model 2: Predict Productivity Change
# ------------------------
st.markdown("## 💻 Model 2: Predict Productivity Change")
with open("model2_productivity_change_bundle.pkl", "rb") as f:
    bundle2 = pickle.load(f)
model2 = bundle2["model"]
features2 = bundle2["selected_features"]
scaler2 = bundle2["scaler"]
numeric2 = ['Age', 'Years_of_Experience', 'Hours_Worked_Per_Week',
            'Number_of_Virtual_Meetings', 'Work_Life_Balance_Rating',
            'Stress_Level', 'Company_Support_for_Remote_Work', 'Physical_Activity_Weekly']
controls2 = {
    'Age': (18, 65), 'Years_of_Experience': (0, 40), 'Hours_Worked_Per_Week': (20, 100),
    'Number_of_Virtual_Meetings': list(range(0, 11)), 'Work_Life_Balance_Rating': [1, 2, 3, 4, 5],
    'Stress_Level': [1, 2, 3, 4, 5], 'Company_Support_for_Remote_Work': [1, 2, 3, 4, 5],
    'Physical_Activity_Weekly': [0, 1]
}
input2 = {}
for col in features2:
    if col in controls2:
        val = controls2[col]
        input2[col] = st.selectbox(col, val, key=f"{col}_model2") if isinstance(val, list) else st.number_input(f"{col} ({val[0]}–{val[1]})", min_value=val[0], max_value=val[1], value=val[0], key=f"{col}_model2")
    else:
        input2[col] = st.selectbox(col, [0, 1], format_func=lambda x: "Yes" if x else "No", key=f"{col}_model2")
if st.button("📈 Predict Productivity"):
    df2 = pd.DataFrame([input2])
    df2[numeric2] = scaler2.transform(df2[numeric2].to_numpy())
    st.success(f"Predicted Productivity Change: {model2.predict(df2[features2])[0]}")

# ------------------------
# Model 3: Predict Satisfaction with Remote Work
# ------------------------
st.markdown("## 😊 Model 3: Predict Satisfaction with Remote Work")
with open("model3_satisfaction_xgb_bundle.pkl", "rb") as f:
    bundle3 = pickle.load(f)
model3 = bundle3["model"]
features3 = bundle3["selected_features"]
scaler3 = bundle3["scaler"]
numeric3 = ['Age', 'Years_of_Experience', 'Hours_Worked_Per_Week',
            'Number_of_Virtual_Meetings', 'Work_Life_Balance_Rating',
            'Social_Isolation_Rating', 'Company_Support_for_Remote_Work']

controls3 = {
    'Age': (18, 65),
    'Years_of_Experience': (0, 40),
    'Hours_Worked_Per_Week': (20, 100),
    'Number_of_Virtual_Meetings': list(range(0, 11)),
    'Work_Life_Balance_Rating': [1, 2, 3, 4, 5],
    'Social_Isolation_Rating': [1, 2, 3, 4, 5],
    'Company_Support_for_Remote_Work': [1, 2, 3, 4, 5],
}


input3 = {}
for col in features3:
    if col in controls3:
        val = controls3[col]
        input3[col] = st.selectbox(col, val, key=f"{col}_model3") if isinstance(val, list) else st.number_input(f"{col} ({val[0]}–{val[1]})", min_value=val[0], max_value=val[1], value=val[0], key=f"{col}_model3")
    else:
        input3[col] = st.selectbox(col, [0, 1], format_func=lambda x: "Yes" if x else "No", key=f"{col}_model3")
if st.button("📊 Predict Satisfaction"):
    df3 = pd.DataFrame([input3])
    missing = [col for col in numeric3 if col not in df3.columns]
    if missing:
        st.error(f"Missing numeric columns: {missing}")
    else:
        df3[numeric3] = scaler3.transform(df3[numeric3].to_numpy())
        st.success(f"Predicted Satisfaction Level: {model3.predict(df3[features3])[0]}")





































