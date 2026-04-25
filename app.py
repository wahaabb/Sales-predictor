# -------------------------------------
#       Imports
# -------------------------------------

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------------
#       Load Data
# -------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("Marketing_Data.csv")

data = load_data()

# -------------------------------------
#       Prepare Data
# -------------------------------------

X = data[["youtube", "facebook", "newspaper"]]
y = data["sales"]

# Train/Test Split (IMPORTANT for real ML)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------
#       Train Model
# -------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------------
#       Evaluate Model
# -------------------------------------

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

coef = model.coef_
intercept = model.intercept_

# -------------------------------------
#       Streamlit UI
# -------------------------------------

st.set_page_config(page_title="Sales Predictor", page_icon="📊", layout="wide")

st.title("📊 AI-Powered Sales Predictor")
st.write("Predict sales based on marketing budget using Machine Learning.")

# Sidebar Inputs
st.sidebar.header("🎯 Adjust Marketing Budget")

youtube = st.sidebar.slider("📺 YouTube", float(X.youtube.min()), float(X.youtube.max()), float(X.youtube.mean()))
facebook = st.sidebar.slider("📘 Facebook", float(X.facebook.min()), float(X.facebook.max()), float(X.facebook.mean()))
newspaper = st.sidebar.slider("📰 Newspaper", float(X.newspaper.min()), float(X.newspaper.max()), float(X.newspaper.mean()))

# Prediction
input_data = [[youtube, facebook, newspaper]]
prediction = model.predict(input_data)[0]

# -------------------------------------
#       Output Section
# -------------------------------------

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("📺 YouTube", f"${youtube:.2f}")
col2.metric("📘 Facebook", f"${facebook:.2f}")
col3.metric("📰 Newspaper", f"${newspaper:.2f}")

st.markdown("---")

st.subheader("📈 Predicted Sales")
st.success(f"💰 Expected Sales: {round(prediction)} units")

# -------------------------------------
#       Model Performance
# -------------------------------------

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("MAE", f"{mae:.2f}")

st.info("R² shows how well the model explains the data. MAE shows average prediction error.")

# -------------------------------------
#       Feature Importance
# -------------------------------------

st.subheader("📌 Feature Impact")

coef_df = pd.DataFrame({
    "Feature": ["YouTube", "Facebook", "Newspaper"],
    "Impact": coef
}).sort_values(by="Impact", ascending=False)

st.bar_chart(coef_df.set_index("Feature"))

st.write("Higher values indicate stronger influence on sales.")

# -------------------------------------
#       Footer
# -------------------------------------

st.markdown("---")
st.caption("🚀 Built with Machine Learning & Streamlit")