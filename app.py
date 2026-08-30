import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Page settings
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon=" ",
    layout="centered"
)

st.title(" Student Performance Prediction")
st.write("Enter the student details to predict the Stress Level.")

# Load dataset
try:
    df = pd.read_csv("students performance.csv")

    # Remove unwanted spaces from column names
    df.columns = df.columns.str.strip()

    # Target column
    target = "Stress_Level"

    if target not in df.columns:
        st.error("Performance Index column not found in the dataset.")
        st.write("Available columns:", list(df.columns))
        st.stop()

    # Separate input and output
    X = df.drop(columns=[target])
    y = df[target]

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "num",
                "passthrough",
                numerical_columns
            )
        ]
    )

    # Machine Learning model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]
    )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    st.subheader(" Enter Student Details")

    user_input = {}

    # Create input boxes automatically based on dataset column
for column in X.columns:

    if column in categorical_columns:
        options = df[column].dropna().unique().tolist()
        user_input[column] = st.selectbox(
            column,
            options
        )

    else:
        min_value = float(df[column].min())
        max_value = float(df[column].max())
        mean_value = float(df[column].mean())

        user_input[column] = st.number_input(
            column,
            min_value=min_value,
            max_value=max_value,
            value=mean_value
        )
        )

    else:
        min_value = float(df[column].min())
        max_value = float(df[column].max())
        mean_value = float(df[column].mean())

        user_input[column] = st.number_input(
            column,
            min_value=min_value,
            max_value=max_value,
            value=mean_value
        )
        )

        user_input[column] = st.number_input(
            column,
            min_value=min_value,
            max_value=max_value,
            value=mean_value
        )

    # Prediction button
    if st.button(" Predict Performance"):

        input_df = pd.DataFrame([user_input])

        prediction = model.predict(input_df)[0]

        st.success(
            f" Predicted Performance Index: {prediction:.2f}"
        )

        if prediction >= 80:
            st.balloons()
            st.write("Excellent Performance!")

        elif prediction >= 60:
            st.write(" Good Performance!")

        elif prediction >= 40:
            st.write(" Average Performance. Keep improving!")

        else:
            st.write(" Need more improvement. Keep studying!")

except FileNotFoundError:
    st.error(
        " Dataset not found. Make sure "
        "'students performance.csv' is uploaded in the repository."
    )

except Exception as e:
    st.error(f"Something went wrong: {e}")
