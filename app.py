import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# -------------------------------
# PAGE SETTINGS
# -------------------------------

st.set_page_config(
    page_title="Student Stress Level Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Stress Level Prediction")
st.write("Enter the student details to predict the Stress Level.")


# -------------------------------
# LOAD DATASET
# -------------------------------

try:

    df = pd.read_csv("students performance.csv")

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Remove extra spaces from text values
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].str.strip()

    # Target column
    target = "Stress_Level"

    # Check target column
    if target not in df.columns:

        st.error("Stress_Level column not found in dataset.")

        st.write(
            "Available columns:",
            list(df.columns)
        )

        st.stop()


    # -------------------------------
    # INPUT AND OUTPUT
    # -------------------------------

    X = df.drop(columns=[target])
    y = df[target]


    # -------------------------------
    # IDENTIFY COLUMNS
    # -------------------------------

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()


    # -------------------------------
    # PREPROCESSING
    # -------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "numerical",
                "passthrough",
                numerical_columns
            )
        ]
    )


    # -------------------------------
    # MACHINE LEARNING MODEL
    # -------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ]
    )


    # -------------------------------
    # TRAIN MODEL
    # -------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(
        X_train,
        y_train
    )


    # -------------------------------
    # STUDENT INPUT
    # -------------------------------

    st.subheader("📝 Enter Student Details")

    user_input = {}


    for column in X.columns:

        if column in categorical_columns:

            options = (
                df[column]
                .dropna()
                .unique()
                .tolist()
            )

            user_input[column] = st.selectbox(
                column,
                options
            )

        else:

            min_value = float(
                df[column].min()
            )

            max_value = float(
                df[column].max()
            )

            mean_value = float(
                df[column].mean()
            )

            user_input[column] = st.number_input(
                column,
                min_value=min_value,
                max_value=max_value,
                value=mean_value
            )


    # -------------------------------
    # PREDICTION BUTTON
    # -------------------------------

    if st.button("🔮 Predict Stress Level"):

        input_df = pd.DataFrame(
            [user_input]
        )

        prediction = model.predict(
            input_df
        )[0]

        st.subheader("📊 Prediction Result")

        if prediction == "High":

            st.error(
                "🔴 Stress Level: HIGH"
            )

            st.write(
                "The student may be experiencing a high level of stress."
            )

        elif prediction == "Medium":

            st.warning(
                "🟠 Stress Level: MEDIUM"
            )

            st.write(
                "The student has a moderate level of stress."
            )

        elif prediction == "Low":

            st.success(
                "🟢 Stress Level: LOW"
            )

            st.write(
                "The student has a low level of stress."
            )

        else:

            st.info(
                f"Predicted Stress Level: {prediction}"
            )


# -------------------------------
# ERROR HANDLING
# -------------------------------

except FileNotFoundError:

    st.error(
        "❌ Dataset not found."
    )

    st.write(
        "Make sure 'students performance.csv' "
        "is uploaded in the same folder as app.py."
    )


except Exception as e:

    st.error(
        f"❌ Something went wrong: {e}"
    )
