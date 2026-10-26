
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import io
from PIL import Image
from datetime import datetime

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Paddy Disease Detection",
    page_icon=None,
    layout="wide"
)

st.title("Paddy Plant Disease Detection")
st.write(
    "Upload paddy leaf images to classify possible diseases "
    "using a trained deep learning model."
)

# ---------------- CLASS NAMES ----------------

class_names = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro"
]

# ---------------- DISEASE INFORMATION ----------------

disease_info = {
    "bacterial_leaf_blight": {
        "Symptoms": "Yellowing, drying and blighting of leaf edges.",
        "Cause": "Bacterial infection, often favored by warm, wet conditions.",
        "Prevention": "Use healthy seed, resistant varieties where suitable, "
                      "and recommended field sanitation practices."
    },
    "bacterial_leaf_streak": {
        "Symptoms": "Narrow, translucent streaks that may turn yellow or brown.",
        "Cause": "Bacterial infection affecting leaf tissues.",
        "Prevention": "Use healthy seed, maintain field hygiene and seek "
                      "local agricultural advice."
    },
    "bacterial_panicle_blight": {
        "Symptoms": "Discoloration or poor grain development around the panicle.",
        "Cause": "Bacterial disease associated with affected rice panicles.",
        "Prevention": "Use healthy seed and follow locally recommended "
                      "crop management practices."
    },
    "blast": {
        "Symptoms": "Spindle-shaped leaf lesions with pale centers and darker edges.",
        "Cause": "Fungal infection favored by suitable moisture and weather conditions.",
        "Prevention": "Use resistant varieties where available and follow "
                      "recommended nutrient and disease management practices."
    },
    "brown_spot": {
        "Symptoms": "Brown, oval or circular spots on leaves.",
        "Cause": "Fungal disease that can be associated with stressed plants.",
        "Prevention": "Use healthy seed, maintain balanced nutrition and "
                      "follow recommended field management."
    },
    "dead_heart": {
        "Symptoms": "The central shoot of a young rice tiller dries and can be pulled out.",
        "Cause": "Often associated with stem-boring insect damage.",
        "Prevention": "Inspect plants for insect damage and follow integrated "
                      "pest management recommendations."
    },
    "downy_mildew": {
        "Symptoms": "Pale or yellow streaks and abnormal leaf growth may occur.",
        "Cause": "Disease-like symptoms can be associated with downy mildew infection.",
        "Prevention": "Use healthy seed, monitor affected plants and obtain "
                      "a local expert diagnosis."
    },
    "hispa": {
        "Symptoms": "White feeding streaks or scraped patches on leaves.",
        "Cause": "Rice hispa beetles feed on leaf surfaces.",
        "Prevention": "Monitor for beetles and leaf damage and follow "
                      "integrated pest management advice."
    },
    "normal": {
        "Symptoms": "No disease class was identified by the model.",
        "Cause": "The image was classified as normal by the model.",
        "Prevention": "Continue regular crop monitoring. A normal prediction "
                      "does not guarantee the plant is disease-free."
    },
    "tungro": {
        "Symptoms": "Yellow-orange discoloration, stunting and reduced growth may occur.",
        "Cause": "Rice tungro is caused by viruses transmitted by leafhoppers.",
        "Prevention": "Monitor leafhoppers, use recommended resistant varieties "
                      "where available and follow local agricultural guidance."
    }
}

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("paddy_disease_model.keras")

try:
    model = load_model()
except Exception as error:
    st.error(
        "Could not load the trained model. Make sure "
        "paddy_disease_model.keras is in the project folder."
    )
    st.stop()

# ---------------- SESSION STORAGE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_results" not in st.session_state:
    st.session_state.latest_results = []

# ---------------- IMAGE UPLOAD ----------------

st.subheader("1. Upload Paddy Leaf Images")

uploaded_files = st.file_uploader(
    "Choose one or more leaf images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.caption(f"{len(uploaded_files)} image(s) selected.")

if st.button("Analyze Images", type="primary",
             disabled=not uploaded_files):

    new_results = []

    progress = st.progress(0)
    status = st.empty()

    for index, uploaded_file in enumerate(uploaded_files):
        try:
            image_bytes = uploaded_file.getvalue()

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            resized_image = image.resize((224, 224))

            image_array = np.asarray(
                resized_image, dtype=np.float32
            )

            image_array = np.expand_dims(image_array, axis=0)

            # Predict disease class
            probabilities = model.predict(
                image_array, verbose=0
            )[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_class = class_names[predicted_index]
            confidence = float(probabilities[predicted_index] * 100)

            result = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": uploaded_file.name,
                "predicted_class": predicted_class,
                "confidence": round(confidence, 2),
                "probabilities": {
                    name: round(float(probabilities[i] * 100), 2)
                    for i, name in enumerate(class_names)
                },
                "image_bytes": image_bytes
            }

            new_results.append(result)
            st.session_state.history.append(result)

        except Exception as error:
            st.warning(
                f"Could not process {uploaded_file.name}: {error}"
            )

        progress.progress((index + 1) / len(uploaded_files))
        status.write(f"Processed {index + 1} of {len(uploaded_files)} images")

    st.session_state.latest_results = new_results
    status.write("Analysis complete.")

# ---------------- SHOW PREDICTIONS ----------------

if st.session_state.latest_results:

    st.divider()
    st.subheader("2. Prediction Results")

    for number, result in enumerate(
        st.session_state.latest_results, start=1
    ):
        st.markdown(f"### Image {number}: {result['filename']}")

        left, right = st.columns([1, 2])

        with left:
            st.image(
                result["image_bytes"],
                caption="Uploaded leaf image",
                use_container_width=True
            )

        with right:
            st.write("**Predicted class:**",
                     result["predicted_class"].replace("_", " ").title())

            st.metric("Model confidence", f"{result['confidence']:.2f}%")

            info = disease_info[result["predicted_class"]]

            st.write("**Possible symptoms:**", info["Symptoms"])
            st.write("**Possible cause:**", info["Cause"])
            st.write("**Prevention / management:**", info["Prevention"])

            with st.expander("View probabilities for all 10 classes"):
                probability_df = pd.DataFrame({
                    "Disease class": list(result["probabilities"].keys()),
                    "Probability (%)": list(result["probabilities"].values())
                })

                probability_df = probability_df.sort_values(
                    "Probability (%)", ascending=False
                )

                st.bar_chart(
                    probability_df.set_index("Disease class")
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

        st.divider()

# ---------------- DASHBOARD ----------------

st.subheader("3. Prediction Dashboard")

history = st.session_state.history

if history:
    total_predictions = len(history)
    unique_classes = len(
        set(item["predicted_class"] for item in history)
    )
    average_confidence = np.mean(
        [item["confidence"] for item in history]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Images analyzed", total_predictions)
    col2.metric("Different predicted classes", unique_classes)
    col3.metric("Average confidence", f"{average_confidence:.2f}%")

    history_df = pd.DataFrame([
        {
            "Time": item["time"],
            "Filename": item["filename"],
            "Predicted class": item["predicted_class"],
            "Confidence (%)": item["confidence"]
        }
        for item in history
    ])

    st.write("**Prediction history**")
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download prediction history (CSV)",
        data=csv_data,
        file_name="paddy_prediction_history.csv",
        mime="text/csv"
    )

    if st.button("Clear prediction history"):
        st.session_state.history = []
        st.session_state.latest_results = []
        st.rerun()

else:
    st.info("Upload and analyze images to populate the dashboard.")

