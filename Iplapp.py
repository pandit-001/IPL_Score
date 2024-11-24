import streamlit as st
import pandas as pd
import pickle
import base64

# Load the model
@st.cache_resource
def load_model(file_path):
    return pickle.load(open(file_path, 'rb'))

# Load the model (adjust the path to your model file)
model = load_model("C:\\Users\\akhil\\OneDrive\\Desktop\\SML_Project\\ipl_score.pkl")

# Background Utility Functions
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_path):
    bin_str = get_base64_of_bin_file(image_path)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

ipl_teams = [
    'Chennai Super Kings', 
    'Delhi Daredevils',  
    'Gujarat Titans',   
    'Kolkata Knight Riders', 
    'Kings XI Punjab',   
    'Mumbai Indians', 
    'Rajasthan Royals', 
    'Royal Challengers Bangalore', 
    'Sunrisers Hyderabad'
]

ipl_cities = [
    'Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 
    'Jaipur', 'Kolkata', 'Mumbai', 'Pune', 'Rajkot', 'Indore'
]

def ipl_score_prediction():
    set_background('4.jpg')  # Set background image
    st.title("IPL Score Predictor")

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('Select Batting Team', sorted(ipl_teams))
    with col2:
        bowling_team = st.selectbox('Select Bowling Team', sorted(ipl_teams))

    col3, col4 = st.columns(2)
    with col3:
        city = st.selectbox('Select City', sorted(ipl_cities))
    with col4:
        overs = st.number_input('Overs Completed', min_value=5.0, max_value=20.0, step=0.1)

    col5, col6 = st.columns(2)
    with col5:
        runs = st.number_input('Runs Scored So Far', min_value=0, step=1)
    with col6:
        wickets = st.number_input('Wickets Lost', min_value=0, max_value=10, step=1)

    target = st.number_input('Target Score', min_value=runs + 1, step=1)  # Ensure target > runs
    runs_last_5 = st.number_input('Runs Scored in Last 5 Overs', min_value=0, step=1)
    wickets_last_5 = st.number_input('Wickets Lost in Last 5 Overs', min_value=0, max_value=10, step=1)

    # Predict Score
    if st.button("Predict Score"):
        if overs < 5:
            st.error("Overs must be greater than or equal to 5.")
        else:
            # Calculate additional features
            balls_left = int((20 - overs) * 6)
            runs_left = target - runs
            wickets_left = 10 - wickets
            crr = runs / overs
            rrr = runs_left / (balls_left / 6) if balls_left > 0 else 0

            # Ensure categories match model's training
            known_teams = model.named_steps['preprocessor'].named_transformers_['onehot'].categories_[0]  # Teams known to the model
            if batting_team not in known_teams:
                st.warning(f"Unknown team: {batting_team}. Replacing with 'Unknown Team'.")
                batting_team = "Unknown Team"
            if bowling_team not in known_teams:
                st.warning(f"Unknown team: {bowling_team}. Replacing with 'Unknown Team'.")
                bowling_team = "Unknown Team"

            # Create input DataFrame for prediction
            input_data = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'overs': [overs],
                'runs': [runs],
                'wickets': [wickets],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'crr': [crr],
                'rrr': [rrr],
                'total_runs_x': [target],  # Target score
                'runs_last_5': [runs_last_5],
                'wickets_last_5': [wickets_last_5]
            })

            try:
                predicted_score = model.predict(input_data)[0]
                st.success(f"Predicted Score: {int(predicted_score)}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if st.button("Back"):
        st.experimental_set_query_params(page="welcome")




# Navigation Logic
def main():
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", ["ipl_score"])[0]

    if page == "ipl_score":
        ipl_score_prediction()

if __name__ == "__main__":
    main()
