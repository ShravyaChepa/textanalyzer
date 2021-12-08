from pickle import STOP
from nltk import text
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import streamlit as st
import pandas as pd
import numpy as np
import math
from streamlit.proto.Markdown_pb2 import Markdown
from gingerit.gingerit import GingerIt
from PIL import Image
from fpdf import FPDF
import base64
from nltk.corpus import wordnet
# import nltk stopwords to remove articles, preposition and other non actionable words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
# Lemmatizer helps to reduce words to the base form
from nltk.stem import WordNetLemmatizer
# Ngrams allows to group words in common pairs or trigrams etc
from nltk import ngrams
from PyDictionary import PyDictionary
import time
# visual library
import matplotlib.pyplot as plt
import seaborn as sns
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
# Counter to count objects
from collections import Counter
# for readability scores
import textstat

from transcribe import *


# some methods

# function to clean text and create data frames for word counts (Know your text section)
def word_frequency(all_text):
    # create tokens, remove numbers and lemmatize words
    new_tokens = word_tokenize(all_text)
    new_tokens = [t.lower() for t in new_tokens]
    new_tokens = [t for t in new_tokens if t not in stopwords.words('english')]
    new_tokens = [t for t in new_tokens if t.isalpha()]

    lemmatizer = WordNetLemmatizer()
    new_tokens = [lemmatizer.lemmatize(t) for t in new_tokens]



    #count the words, pairs and trigrams
    count1 = Counter(new_tokens)
    count2 = Counter(ngrams(new_tokens,2))
    count3 = Counter(ngrams(new_tokens,3))

    # create 3 dataframes and return them
    word_freq = pd.DataFrame(count1.items(), columns=['word','frequency']).sort_values(by='frequency', ascending=False)
    word_pairs = pd.DataFrame(count2.items(), columns=['pairs','frequency']).sort_values(by='frequency',ascending=False)
    trigrams =pd.DataFrame(count3.items(),columns=['trigrams','frequency']).sort_values(by='frequency',ascending=False)

    return word_freq, word_pairs, trigrams

def calculate_time(word_count, rate):
    calc_time = word_count/rate
        # to separate the numerical and decimal part of the number
    modified_calc_time = math.modf(calc_time)
    in_mins = int(modified_calc_time[1])
    in_seconds = round(modified_calc_time[0] * 60, 2)

    display_calc_time = pd.DataFrame(
        [[word_count, in_mins, in_seconds]],
        columns=["Total words","Mins","Seconds"]
        )
    st.table(display_calc_time.style)


#global styles
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #FFF9B6;
}
div.stButton > button:first-child:hover {
    background-color: #FFF;
}
</style>""", unsafe_allow_html=True)


#heading
image = Image.open('note.png')
st.image(image, use_column_width=False, width=100)

st.write("""# Text Analyzer 
A tool to make your essays, speeches, stories, poems, papers and reports better!
***
""")

#sidebar (menu)
st.sidebar.image(image,use_column_width=False,width=70)
st.sidebar.title("Menu")
nav_button = st.sidebar.radio("", (
                                  'Speech to text',
                                  'Know your text',
                                  'Time your text',
                                  'Readability scores',
                                  'Grammar and spellings',
                                  'Definitions',
                                  'Synonyms and antonyms', 
                                  'Title generation',
                                  'Text summarization',
                                  'Fact checking and Plagiarism',
                                  'Download PDF'
                                  )) 

# text input displayed in all pages
with st.expander("Your text"):
    usertext_input = st.text_area(label="", height=400)



# speech to text conversion using AssemblyAI
if nav_button == 'Speech to text': 
    
    file_object = st.file_uploader(label="Please upload your audio file")

    if st.button("transcribe") :

        if file_object:
            token, t_id = upload_file(file_object)
            result = {}
            # polling
            sleep_duration = 1
            percent_complete = 0
            progress_bar = st.progress(percent_complete)
            st.text("Currently in queue")

            while result.get("status") != "processing":
                percent_complete += sleep_duration
                time.sleep(sleep_duration)
                progress_bar.progress(percent_complete/10)
                result = get_text(token, t_id)
                sleep_duration = 0.01
                for percent in range(percent_complete, 101):
                    time.sleep(sleep_duration)
                    progress_bar.progress(percent)

            # polling again until the result status changes to completed
            with st.spinner("Processing....."):
                while result.get("status") != 'completed':
                    result = get_text(token,t_id)

            st.write("#### Transcribed text")
            st.write("""
            Copy and paste this into 'Your text' field and continue 
            ***
            """)
            st.markdown(result['text'])

        else:
            st.text("Please upload a file")

        
# Know your text
elif nav_button == 'Know your text': 
    st.write("##### Number of words and characters")
    if st.button("show"):
        num_of_chars = len(usertext_input)
        num_of_words = len(usertext_input.split())
        st.text("Number of characters: "+ str(num_of_chars))
        st.text("Number of words: "+ str(num_of_words))

    st.write("##### Most frequent words, pairs and trigrams")
    colu1, colu2, colu3 = st.columns(3)
    with colu1:
        num_of_freq_words = st.number_input(label="Number of frequent words",min_value=1,max_value=20)
    with colu2:
        num_of_pairs = st.number_input(label="Number of pairs",min_value=1,max_value=20)
    with colu3:
        num_of_tris = st.number_input(label="Number of trigrams",min_value=1,max_value=20)    
    

    if st.button(label="stats"):
        if usertext_input == '':
            st.text("Please enter some text")
        else:
            data1 , data2, data3 = word_frequency(usertext_input)
            data1.reset_index(inplace = True, drop = True)
            data2.reset_index(inplace = True, drop = True)
            data3.reset_index(inplace = True, drop = True)
            col1, col2= st.columns(2)
            with col1:
                st.write(data1.head(num_of_freq_words))
            with col2:
                st.write(data2.head(num_of_pairs))
            st.write(data3.head(num_of_tris))

            st.text("")        

            #create subplot for the different dataframes of frequency
            fig, axes = plt.subplots(3,1,figsize=(8,15))
            sns.barplot(ax=axes[0],x='frequency',y='word',data=data1.head(num_of_freq_words))
            sns.barplot(ax=axes[1],x='frequency',y='pairs',data=data2.head(num_of_pairs))
            sns.barplot(ax=axes[2],x='frequency',y='trigrams',data=data3.head(num_of_tris))
            st.pyplot(fig)

        
# time your text section
elif nav_button == 'Time your text':
    st.write("#### Reading time")
    if st.button("compute"):
        if usertext_input == '':
            st.text("Please enter some text")
        else:
            word_count = len(usertext_input.split())
            calculate_time(word_count, 200)
        
    st.write("#### Speaking time")
    speed_speech = st.radio("Choose your speaking speed",('slow','moderate','fast'))
    if st.button("calculate"):
        if usertext_input == '':
            st.text("Please enter some text")
        else:
            if speed_speech == 'slow':
                word_count = len(usertext_input.split())
                calculate_time(word_count, 110)
            if speed_speech == 'moderate':
                word_count = len(usertext_input.split())
                calculate_time(word_count, 140)
            if speed_speech == 'fast':
                word_count = len(usertext_input.split())
                calculate_time(word_count, 170)
        
elif nav_button == 'Readability scores':
    st.write("#### Readability index")
    # display information about various readability standards
    c1, c2 = st.columns(2)
    with c1:
        score_type1 = st.checkbox("Flesch reading ease")
    with c2:
        st.write("It can be used to assess the ease of readability of any general document.")

    c3, c4 = st.columns(2)
    with c3:
        score_type2 = st.checkbox("Gunning fog index")
    with c4:
        st.write("It is ideal for education material aimed at business houses like magazines and journals.")

    c5, c6 = st.columns(2)
    with c5:
        score_type3 = st.checkbox('Dale-Chall readability score')
    with c6:
        st.write("It uses a lookup table of the most commonly used 3000 English words. It can be used as a grade formula. For example a score of 9.3 means that a ninth grader would be able to read the document.")
    
    #compute and display readability scores
    if st.button(label="compute"):
        if usertext_input == "":
            st.text("Please enter some text")
        else:
            if score_type1:
                st.markdown("**Flesch reading ease**")
                score = textstat.flesch_reading_ease(usertext_input)
                st.text("Readability score: "+ str(score))
                if score >= 90 :
                    st.text("Difficulty: Very easy")
                elif 80 <= score < 90:
                    st.text("Difficulty: Easy")
                elif 70 <= score < 80:
                    st.text("Difficulty: Fairly easy")
                elif 60 <= score < 70:
                    st.text("Difficulty: Standard")
                elif 50 <= score < 60:
                    st.text("Difficulty: Fairly hard")
                elif 30 <= score < 50:
                    st.text("Difficulty: Hard")
                else:
                    st.text("Difficulty: Very hard")

            if score_type2:
                st.markdown("**Gunning fog index**")
                score = round(textstat.gunning_fog(usertext_input))
                st.text("Readability score: "+ str(score))
                if score >= 20:
                    st.text("Academic level: Post-graduate plus")
                elif 17 <= score < 20:
                    st.text("Academic level: Post-graduate")
                elif 16 <= score < 17:
                    st.text("Academic level: College senior")
                elif 13 <= score <= 15:
                    st.text("Academic level: College junior, sophomore, freshman")
                elif 11 <= score <= 12:
                    st.text("Academic level: High school senior, junior")
                elif 9 < score <= 10:
                    st.text("Academic level: High school sophomore")
                elif 8 < score <= 9:
                    st.text("Academic level: High school freshman")
                elif 7 < score <= 8:
                    st.text("Academic level: 8th grade")
                elif 6 < score <= 7:
                    st.text("Academic level: 7th grade")
                else:
                    st.text("Academic level: 6th grade or lower")
                
            if score_type3:
                st.markdown("**Dale-Chall readability score**")
                score = round(textstat.dale_chall_readability_score(usertext_input), 2)
                st.text("Readability score: "+ str(score))
                if 9.0 <= score <= 9.9:
                    st.text("Grade: Easily understood by an average 13th to 15th grade (college) student")
                elif 8.0 <= score <= 8.9:
                    st.text("Grade: Easily understood by an average 11th or 12th grade student")
                elif 7.0 <= score <= 7.9:
                    st.text("Grade: Easily understood by an average 9th or 10th grade student")
                elif 6.0 <= score <= 6.9:
                    st.text("Grade: Easily understood by an average 7th or 8th grade student")
                elif 5.0 <= score <= 5.9:
                    st.text("Grade: Easily understood by an average 5th or 6th grade student")
                else:
                    st.text("Grade: Easily understood by an average 4th grade student or lower")

# for grammar and spelling check

elif nav_button == 'Grammar and spellings':

    grammar_check_text = st.text_area(label="Enter text to check grammar or spellings:")
    parser = GingerIt()
    if st.button('check grammar'):
        if grammar_check_text == '':
            st.text("Please enter some text")
        else: 
            st.markdown("Copy the corrected version of your essay into the textbox above and continue!")
            grammar_corrected_dict = parser.parse(grammar_check_text)
            st.text_area(label="Modified text: ", value= str(grammar_corrected_dict["result"]), height=400)
            st.markdown("Number of corrections made: " + str(len(grammar_corrected_dict['corrections'])))


# for definitions of user given word
elif nav_button == 'Definitions': 

    define_word = st.text_input(label="Enter a word")
    dict = PyDictionary()
    if st.button(label="define"):
        if define_word == "":
            st.text("Please enter some text")
        else:
            definitions = dict.meaning(define_word)
            st.write(definitions)
        

# for synonyms and antonyms of user given word
elif nav_button == 'Synonyms and antonyms': 

    sa_word = st.text_input(label="Enter a word:")
 
    if st.button("synonyms and antonyms"):
        if usertext_input == '':
            st.text("Please enter some text")
        else:
            synonyms = []
            for syn in wordnet.synsets(sa_word):
                for lm in syn.lemmas():
                    synonyms.append(lm.name())
            st.write("Synonyms:")
            st.markdown(synonyms)
            antonyms = []
            for syn in wordnet.synsets(sa_word):
                for lm in syn.lemmas():
                    if lm.antonyms():
                        antonyms.append(lm.antonyms()[0].name())
            st.write("Antonyms:")
            st.markdown(antonyms)

# title generation   
elif nav_button == 'Title generation':
    st.write("###### Generate the most important sentences of your text to help you come up with a title.")
    no_of_imp_sentences = st.number_input(label="Number of sentences",min_value=3, max_value=10,step=1)
    if st.button(label="generate"):
        if usertext_input == '':
            st.text("Please enter some text")
        else:
            parser = PlaintextParser.from_string(usertext_input, Tokenizer('english'))

            #creating the summarizer
            lsa_summarizer = LsaSummarizer()
            lsa_summary = lsa_summarizer(parser.document, no_of_imp_sentences)

            for sentence in lsa_summary:
                st.write(sentence)
    

# text summarization
elif nav_button == 'Text summarization':
    no_of_sentences = st.number_input(label="Number of sentences",min_value=2, max_value=10,step=1)
    if st.button(label="summarize"):

        if usertext_input == '':
            st.text("Please enter some text")
        else:

#             textrank_summary = ''
#             #tokenizing the text
#             stop_words = set(stopwords.words("english")) 
#             words = word_tokenize(usertext_input)
#             # creating a frequency table to keep the score of each word
#             freq_table = dict()
#             for word in words:
#                 word = word.lower()
#                 if word in stop_words:
#                     continue
#                 if word in freq_table:
#                     freq_table[word] +=1
#                 else:
#                     freq_table[word] = 1

#             #creating a dictionary to keep the score of each sentence
#             sentences = sent_tokenize(usertext_input)
#             sentence_value = dict()

#             for sentence in sentences:
#                 for word, freq in freq_table.items():
#                     if word in sentence.lower():
#                         if sentence in sentence_value:
#                             sentence_value[sentence] += freq
#                         else:
#                             sentence_value[sentence] = freq

#             sum_values = 0
#             for sentence in sentence_value:
#                 sum_values += sentence_value[sentence]


#             # average value of a sentence from the original text

#             average = int(sum_values/len(sentence_value))

#             #storing sentences into our summary
#             for sentence in sentences:
#                 if ( sentence in sentence_value) and (sentence_value[sentence] > (1.2 * average)):
#                     textrank_summary += " " + sentence
#                     st.write(textrank_summary)
            
#             st.write(textrank_summary)

#             if textrank_summary == "":

                #use lsa alg
              
              parser = PlaintextParser.from_string(usertext_input, Tokenizer('english'))

                #creating the summarizer
              lsa_summarizer = LsaSummarizer()
              lsa_summary = lsa_summarizer(parser.document, no_of_sentences)

              for sentence in lsa_summary:
                  st.write(sentence)

            

# for fact checking and plagiarism
elif nav_button == 'Fact checking and Plagiarism': 
    st.write("#### Fact Checking")
    st.write("Check out the Fact Check Explorer by Google [here](https://toolbox.google.com/factcheck/explorer)")

    st.write("#### Plagiarism Checking")
    st.write("Check out this plagiarism detector [here](https://plagiarismdetector.net/)")
    

else:
    st.write('Make sure the desired version of your text is present in the textbox and click the download button to download a pdf file')
    save_file_name = st.text_input(label="Name file")
    export_as_pdf = st.button(label="export as PDF")

    def create_download_link(val, filename):
        b64 = base64.b64encode(val)  # val looks like b'...'
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download {filename}.pdf</a>'

    if export_as_pdf:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', '' , 8)
            pdf.set_xy(10.0,10.0)
            pdf.multi_cell(190, 5, usertext_input)
        
            html = create_download_link(pdf.output(dest="S").encode("latin-1"), save_file_name)

            st.markdown(html, unsafe_allow_html=True)
        except:
            st.text("Something went wrong. Please try again.")
        

