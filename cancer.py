import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle


st.header("Breast Cancer Treatment Decisions")


# Load the trained model
loaded_model = pickle.load(open('model_for_Cancer_Irradiat.pickle', 'rb'))

data = pd.read_csv("D:\\DOWNLOAD\\Train_Breast_Cancer_csv.csv")


# Define mappings for categorical variables
class_map = {'no-recurrence-events': 0, 'recurrence-events': 1}
age_map = {'20-29': 0, '30-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70-79': 5}
menopause_map = {'premeno': 0, 'ge40': 1, 'lt40': 2}
tumor_size_map = {'0-4': 0, '5-9': 1, '10-14': 2, '15-19': 3, '20-24': 4, '25-29': 5, '30-34': 6, '35-39': 7, '40-44': 8, '45-49': 9, '50-54': 10}
inv_nodes_map = {'0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-17': 5, '24-26': 6}
node_caps_map = {'no': 0, 'yes': 1}
deg_malig_map = {'1': 1, '2': 2, '3': 3}
breast_map = {'left': 0, 'right': 1}
breast_quad_map = {'left_low': 0, 'left_up': 1, 'right_low': 2, 'right_up': 3, 'central': 4}

# Custom colors
# header_bg_color = "#f0f0f0"
# header_text_color = "#008080"
intro_bg_color = "#ccffcc"


# Add an introduction
st.markdown(
    f"""
    <div style="background-color:{intro_bg_color};padding:40px;border-radius:10px">
        <p style="color:black;text-align:justify;font-size:20px;">
        Breast cancer is an abnormal development of malignant cells. The cancer spreads to other parts of the body if untreated. 
        It is important to stop the spread of breast cancer at the initial stage to prevent damage of other parts. 
        However, due to many factors it attacks the lymph node and affects the immune system resulting in its multiplication. 
        Here, we will classify the people who needs Radiation Therapy or not via the attributes 
        age, class, menopause, tumor size, inv nodes,node-caps, degree of malignancy, breast, breast quadrant.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input fields for categorical variables
CLASS = st.selectbox('Class(patient experienced a recurrence of breast cancer or not after the initial treatment)', ['no-recurrence-events', 'recurrence-events'])
AGE = st.selectbox('Age(Age of the patient at the time of diagnosis)', ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79'])
MENOPAUSE = st.selectbox('Menopause', ['premeno', 'ge40', 'lt40'])
TUMORSIZE = st.selectbox('Tumor-size( size of the cancer tumor at the time of diagnosis)', ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54'])
INVNODES = st.selectbox('Inv-nodes', ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '24-26'])
NODECAPS = st.selectbox('Node-caps', ['no', 'yes'])
DEGMALIG = st.selectbox('Deg-malig(Grade of cancer that is visible under a microscope)', ['1', '2', '3'])
BREAST = st.selectbox('Breast(side of the breast)', ['left', 'right'])
BREASTQUAD = st.selectbox('Breast-quad(nipple area breast cancer occurred)', ['left_low', 'left_up', 'right_low', 'right_up', 'central'])




if st.button('Predict', key='predict_button'):
    st.markdown(
        """
        <style>
        .predict-button {
            color: white;
            background-color: #4CAF50; /* Green */
            font-size: 26px;
            padding: 15px 28px;
            cursor: pointer;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Transform categorical inputs to encoded values
    class_encoded = class_map[CLASS]
    age_encoded = age_map[AGE]
    menopause_encoded = menopause_map[MENOPAUSE]
    tumor_size_encoded = tumor_size_map[TUMORSIZE]
    inv_nodes_encoded = inv_nodes_map[INVNODES]
    node_caps_encoded = node_caps_map[NODECAPS]
    deg_malig_encoded = deg_malig_map[DEGMALIG]
    breast_encoded = breast_map[BREAST]
    breast_quad_encoded = breast_quad_map[BREASTQUAD]

    # Combine all inputs into a list for prediction
    user_input = [[class_encoded, age_encoded, menopause_encoded, tumor_size_encoded,
                   inv_nodes_encoded, node_caps_encoded, deg_malig_encoded,
                   breast_encoded, breast_quad_encoded]]

    # Make predictions based on user input
    result = loaded_model.predict(user_input)

    # Display prediction value
    st.write('Prediction:', result[0])

    # Display comment based on prediction value
    if result[0] == 0:
        st.write("Patient needs radiation therapy as a treatment for breast cancer")
    else:
        st.write("Patient in a well condition")
