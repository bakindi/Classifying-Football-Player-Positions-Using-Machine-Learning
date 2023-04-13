import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc
import joblib
from joblib import dump, load
import pickle

#page creation
st.set_page_config(
    page_title="player's position prediction",
    page_icon="https://ceotudent.com/wp-content/uploads/2022/04/futbol-taktikleri.jpg",
    menu_items={"Get Help":"https://github.com/bakindi",
                "About":"for more information"+"https://github.com/bakindi"
                }
)

# Loading the model
model = joblib.load(open('C:/streamlit_dev/finalized_model (1).sav', 'rb'))
# Column names of the csv to be shown to the user as an example.
x = pd.read_csv("C:/streamlit_dev/columns_name.csv")
# Creating a sample CSV file
example_data = pd.DataFrame(np.random.randint(0, 10, size=(1,75)), columns=x.columns)

# UI creation
st.title('Player Position Prediction')
st.markdown("Increase the success of your team by playing your football player in the most suitable position. All you have to do for this is to give the data containing the statistics of your football player to our model that will predict it.")
st.image("https://thekoptimes.com/wp-content/uploads/2021/02/files-fbl-eng-pr-liverpool-man-utd-scaled.jpg")
st.write("Please upload the CSV file with the player's performance data. Below is a sample CSV file.")

# Show sample CSV file
st.dataframe(example_data)

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    # Reading the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Matching the model to the expected data format
    X = np.array(df).reshape(1, -1)
    
    # guessing
    prediction = str(model.predict(X)[0])

    predicts = {"0":'Goalkeeper' ,
                "1":'Defender - Centre-Back',
                "2":'Defender - Right-Back',
                "3":'Defender - Left-Back',
                "4":'Midfielder - Left Midfield',
                "5":'Midfielder - Right Midfield',
                "6":'Midfielder - Defensive Midfield',
                "7":'Forward - Left Winger',
                "8":'Midfielder - Central Midfield',
                "9":'Forward - Centre-Forward',
                "10":'Midfielder - Attacking Midfield',
                "11":'Forward - Right Winger'
    }

    fig, ax = plt.subplots()

    # Draw the green grass
    ax.add_patch(Rectangle((0, 0), 120, 80, color='green'))

    # Draw the penalty area
    ax.add_patch(Rectangle((0, 22.3), 16.5, 35.3, fill=False,color="white"))
    ax.add_patch(Rectangle((103.5, 22.3), 16.5, 35.3, fill=False,color="white"))

    # Draw the goal box
    ax.add_patch(Rectangle((0, 30.7), 5.5, 18.6, fill=False,color="white"))
    ax.add_patch(Rectangle((114.5, 30.7), 5.5, 18.6, fill=False,color="white"))
    # Draw the penalty spot
    ax.add_patch(plt.Circle((11, 40), 0.5, color='white'))
    ax.add_patch(plt.Circle((109, 40), 0.5, color='white'))
    # Draw the penalty arc
    penalty_arc = Arc((11.5, 40), height=16.2, width=16.2, angle=0, theta1=310, theta2=50, color='white')
    penalty_arc_ = Arc((109, 40), height=16.2, width=16.2, angle=180, theta1=310, theta2=50, color='white')
    ax.add_patch(penalty_arc)
    ax.add_patch(penalty_arc_)
    # Add the center circle
    ax.add_patch(plt.Circle((60, 40), 10, color='white', fill=False))

    # Add the center line
    ax.plot([60, 60], [0, 80], color='white')

    # Define the position of each player on the field
    positions = {
        'Goalkeeper': (118, 40),
        'Defender - Centre-Back': (102, 40),
        'Defender - Right-Back': (102, 65),
        'Defender - Left-Back': (102, 15),
        'Midfielder - Defensive Midfield': (80, 40),
        'Midfielder - Central Midfield': (60, 40),
        'Midfielder - Right Midfield': (60, 65),
        'Midfielder - Left Midfield': (60, 15),
        'Midfielder - Attacking Midfield': (30, 40),
        'Forward - Right Winger': (18, 65),
        'Forward - Left Winger': (18, 15),
        'Forward - Centre-Forward': (10, 40)
    }

    # Define the color for each position
    colors = {
        'Goalkeeper': 'blue',
        'Defender - Centre-Back': 'blue',
        'Defender - Right-Back': 'blue',
        'Defender - Left-Back': 'blue',
        'Midfielder - Defensive Midfield': 'yellow',
        'Midfielder - Central Midfield': 'yellow',
        'Midfielder - Right Midfield': 'yellow',
        'Midfielder - Left Midfield': 'yellow',
        'Midfielder - Attacking Midfield': 'red',
        'Forward - Right Winger': 'red',
        'Forward - Left Winger': 'red',
        'Forward - Centre-Forward': 'red'
    }
    # Add the players to the field
    posits = predicts.get(prediction)
    coords = positions.get(posits)
    color = colors[posits]        
    ax.add_patch(plt.Circle(coords, 6, color=color))
    
    # Remove the axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("rival half field")
    ax.set_label("12 different positions in football")
        
    # Show result
    st.write('prediction result:', predicts.get(prediction))
    st.pyplot(fig)


