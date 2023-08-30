import streamlit as st
import pickle
import pandas as pd
import streamlit_toggle.toggle as toggle
# from streamlit_toggle import toggle
import numpy as np
st.header("Mental disorder screener App")
data = pd.read_csv("./datasets/Mental_disorder_symptoms.csv")


# load model
loaded_model = pickle.load(open("MHSC_RF.sav", "rb"))

list_variables = ['age','feeling.nervous','panic','breathing.rapidly','sweating','trouble.in.concentration','having.trouble.in.sleeping','having.trouble.with.work','hopelessness','anger','over.react','change.in.eating','suicidal.thought','feeling.tired','close.friend','social.media.addiction','weight.gain','introvert','popping.up.stressful.memory','having.nightmares','avoids.people.or.activities','feeling.negative','trouble.concentrating','blamming.yourself','hallucinations','repetitive.behaviour','increased.energy']
list_outcomes = ['ADHD','ASD','Loneliness','MDD','OCD','PDD','PTSD','Anxiety','Bipolar','Eating Disorder','Psychotic depression','Sleeping disorder']

if st.checkbox('Show Training Dataframe'):
    data

st.subheader("")
input_age = st.number_input('Age', 0, 100,30)

left_column, right_column = st.columns(2)
with left_column:
    input_nervous = st.toggle('Feeling Nervous',key=2)
    input_panic = st.toggle('Panic', key=3)
    input_breathing_rapidly = st.toggle('Breathing Rapidly',key=4)
    input_sweating = st.toggle('Sweating', key=5)
    input_trouble_concentrating = st.toggle('Trouble Concentrating', key=6)
    input_trouble_sleeping = st.toggle('Trouble Sleeping', key=7)
    input_troublewithwork = st.toggle('Trouble with Work',key=8)
    input_hopelessness = st.toggle('Hopelessness',key=9)
    input_anger = st.toggle('Anger',key=10)
    input_over_react = st.toggle('Over React',key=11)
    input_change_in_eating = st.toggle('Change in Eating',key=12)
    input_suicidal_toughts = st.toggle('Suicidal Thoughts',key=13)
    input_feeling_tired = st.toggle('Feeling Tired',key=14)

with right_column:
    input_close_friend = st.toggle('Have Close Friend', key=15)
    input_social_media = st.toggle('Social Media Addiction',key=16)
    input_weight_gain = st.toggle('Weight Gain',key=17)
    input_introvert = st.toggle('Introvert',key=18)
    input_popping_stress_memory = st.toggle('Popping Up Stressful Memories', key=19)
    input_having_nightmares = st.toggle('Having Nightmares', key=20)
    input_avoid_people_activities = st.toggle('Avoids People or Activities', key=21)
    input_feeling_negative = st.toggle('Feeling Negative', key=22)
    input_trouble_concentrating = st.toggle('Trouble Concentrating', key=23)
    input_blaming_yourself = st.toggle('Blaming Yourself', key=24)
    input_hallucinations = st.toggle('Have Hallucinations', key=25)
    input_repetitive_behavior = st.toggle('Repetitive Behavior', key=26)
    input_increase_energy = st.toggle('Increased Energy', key=27)



test = pd.DataFrame({
    'score': [input_age,input_nervous,input_panic,input_breathing_rapidly,input_sweating,input_trouble_concentrating,input_trouble_sleeping,input_troublewithwork,input_hopelessness,input_anger,input_over_react,input_change_in_eating,input_suicidal_toughts,input_feeling_tired,input_close_friend,input_social_media,input_weight_gain,input_introvert,input_popping_stress_memory,input_having_nightmares,input_avoid_people_activities,input_feeling_negative,input_trouble_concentrating,input_blaming_yourself,input_hallucinations,input_repetitive_behavior,input_increase_energy],
    'column': list_variables
})
test_data= test.set_index('column').transpose()

# test = pd.DataFrame({
#     'score': [True,0,0,False,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
#     'column': list_variables
# })
# test_data= test.set_index('column').transpose()

st.write("")

if st.button('Make Prediction'):
    pred = loaded_model.predict(test_data)[0]
    final_pred = list_outcomes[pred]
    st.header(f"You may or may not have: {final_pred}")
    st.write(f"Thank you. Please don't trust this and check with professional")

st.write("")
st.write("")
st.write("")
st.subheader(f"WARNING: This machine is NOT train on NORMAL dataset, There is NO NORMAL RESULT. Thus this is only for showcasing and learning purposes")
st.subheader(f"The data itself is very questionable")

st.write("")
st.write("")
st.write("created by Ronald Leonardo")
st.write("Dataset by baselbakeer | https://www.kaggle.com/datasets/baselbakeer/mental-disorders-dataset")

