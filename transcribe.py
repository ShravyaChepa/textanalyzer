import os
import requests



# uploading a local audio file to AssemblyAI
def get_url(token, data):
    headers = { 'authorization': token}
    response = requests.post('https://api.assemblyai.com/v2/upload', headers= headers, data=data)
    url = response.json()["upload_url"]
    return url


# uploading a file for transcription
def get_transcribe_id(token, url):
    endpoint = "https://api.assemblyai.com/v2/transcript"
    json = {
        "audio_url": url
    }
    headers = {
        "authorization": token,
        "content_type": "application/json"
    }
    response = requests.post(endpoint, json=json, headers=headers)
    id = response.json()['id'] # for id of the transcription session
    return id

# downloading an audio transcription

# status of transcription changes from "queued" to "processing" to "completed" if there are no errors
# polling: process of making requests and waiting till the status is completed (using a while loop)

def get_text(token, transcribe_id):
    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcribe_id}"
    headers = {
        "authorization": token
    }
    result = requests.get(endpoint, headers= headers).json()
    return result

# The third function will call both of the previous functions successively.

# This function will also be connected to the “Upload” button in our Streamlit UI. The function has only one parameter: the file object. The function will do the following

#     It will load the API token from our st.secrets dictionary.
#     It will use the token to call the previously defined functions
#     It will return the transcription ID

def upload_file(file_obj):
    token = st.secrets["API_TOKEN"]
    file_url = get_url(token,file_obj)
    transcribe_id = get_transcribe_id(token,file_url)
    return token,transcribe_id
