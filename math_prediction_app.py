import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt

# Writing the app title and description
st.write("""
         # Prediction of Mathematic Result
""")
st.write('---')
st.write('This app predicts the Mathematic Result of a student based on their background')
st.write('Based on your input, Your Mathematic Result is : ')

st.sidebar.header('User Input Parameters')

# Get user input for parameters
def user_input_features(): 
    Gender = st.sidebar.selectbox('Gender', ('female', 'male'))
    LunchType = st.sidebar.selectbox('Lunch Type', ('free/reduced', 'standard'))
    TestPrep = st.sidebar.selectbox('Test Preparation Course', ('completed', 'none'))
    ParentEduc = st.sidebar.selectbox('Parental Level of Education', ("associate's degree", "bachelor's degree", 'high school', "master's degree", 'some college', 'some high school'))
    PracticeSport = st.sidebar.selectbox('Practice Sport', ('never', 'regularly', 'sometimes'))
    WklyStudyHours = st.sidebar.selectbox('Weekly Study Hour', ('5 - 10', '< 5', '> 10'))
    WritingScore = st.sidebar.slider('Writing Score', 0, 100, 50)
    ReadingScore = st.sidebar.slider('Reading Score', 0, 100, 50)
    data = {'ReadingScore': ReadingScore, 
            'WritingScore': WritingScore,
            'Gender': Gender, 
            'ParentEduc': ParentEduc, 
            'LunchType': LunchType,
            'TestPrep': TestPrep,
            'PracticeSport': PracticeSport,
            'WklyStudyHours': WklyStudyHours,
            }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Combining user input with entire dataset
data_raw = pd.read_csv("./Result based on student's background.csv")
data = data_raw.drop(columns=['Unnamed: 0', 'MathScore', 'IsFirstChild', 'NrSiblings', 'TransportMeans', 'ParentMaritalStatus', 'EthnicGroup'])
df = pd.concat([input_df,data],axis=0)

# Encoding of input variables
encode = ['Gender', 'ParentEduc', 'LunchType', 'TestPrep', 'PracticeSport', 'WklyStudyHours']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)


# Loading model
load_clf = pickle.load(open('mathematic_result_prediction.pkl', 'rb'))

# Make prediction
prediction = load_clf.predict(df)

# Normalize the result
if(prediction>100): 
    prediction[0] = 100

# Print result
result = np.round(prediction[0], 2)
st.write('''## ''', result, ''' ##''')

# Plotting the result
data = pd.DataFrame({
    'Mathematic Result': prediction, 
    'Your Result': 'Mathematic Result'
})

bars = alt.Chart(data).mark_bar().encode(
    y = alt.Y(
        'Mathematic Result', 
        scale=alt.Scale(domain=[0, 100]), 
    ), 
    x="Your Result", 
).properties(
    width=550
)

# Display bar chart
chart = alt.vconcat(bars,  title="Mathematic Result Prediction")
st.altair_chart(chart, use_container_width=True)

st.write('---')
st.write('Thank you for using my application!')