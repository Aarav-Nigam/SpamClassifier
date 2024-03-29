import streamlit as st
import pickle
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')



def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('models/vectorizer.pkl','rb'))
model = pickle.load(open('models/model.pkl','rb'))

st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    page_icon=":email:",
    layout="centered"
)

# Background image
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://editor.analyticsvidhya.com/uploads/32086heading.jpeg") center center;
        background-size: cover;
        
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Apply custom CSS to change the background color
st.markdown(
    """
    <style>
    #email-sms-spam-classifier {
        background-color: #000000; 
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message",height=150)

# Apply custom CSS to change the font size
st.markdown(
    """
    <style>
    .st-ct {
        font-size: 40px; /* Change this to the desired font size */
    }
    th{
     color:black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button('Predict',type='primary'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam",)
    else:
        st.header("Not Spam")

# Create a sample dataframe
header_style = """
<style>
h2 {
    color: black; 
}
</style>
"""

# Display the header with custom text color
st.markdown(header_style, unsafe_allow_html=True)
st.header("Sample Messages",divider="red")


data = {'SPAM': ["Thanks for your subscription to Ringtone UK your mobile will be charged �5/month Please confirm by replying YES or NO. If you reply NO you will not be charged",
                 "WINNER!! As a valued network customer you have been selected to receivea �900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
                 "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030",
                 "URGENT! You have won a 1 week FREE membership in our �100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"],
        'NOT SPAM': ["Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
                     "Nah I don't think he goes to usf, he lives around here though",
                     "Even my brother is not like to speak with me. They treat me like aids patent.",
                     "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"]}
df = pd.DataFrame(data)


# Display the dataframe with a white background color
st.table(df.style.set_properties(**{'background-color': 'white', 'color':'black'}))

# Define the footer HTML content with CSS for sticky positioning
footer = '''
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
}
</style>
<div class="footer">
    <p>Developed  by <a style='display: block; text-align: center; color:black ;font-weight: bold; font-size:20px' href="https://portfolio-aarav.netlify.app/" target="_blank">~Aarav Nigam</a></p>
</div>
'''

# Display the footer using st.markdown
st.markdown(footer, unsafe_allow_html=True)