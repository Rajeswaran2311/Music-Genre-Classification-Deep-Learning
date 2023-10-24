from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from flask_uploads import UploadSet, configure_uploads, AUDIO
from werkzeug.datastructures import FileStorage
import io
import math
import wave
import librosa
from datetime import datetime
import tempfile
import soundfile as sf
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

import os

app = Flask(__name__)
audio_files = UploadSet('audio', AUDIO)
UPLOAD_FOLDER = 'uploads'  # Directory to save uploaded audio files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('models/MusicGenre_CNN_79.73.h5')
def process_input(audio_blob, track_duration):
    SAMPLE_RATE = 22050
    NUM_MFCC = 13
    N_FTT = 2048
    HOP_LENGTH = 512
    NUM_SEGMENTS = 10

    samples_per_segment = int(SAMPLE_RATE * track_duration / NUM_SEGMENTS)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

    audio_data = audio_blob.read()
    signal, sample_rate = librosa.load(io.BytesIO(audio_data), sr=SAMPLE_RATE)

    mfcc_features = []

    for d in range(NUM_SEGMENTS):
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T
        mfcc_features.append(mfcc.tolist())

    return mfcc_features
def predict_genre(audio_blob):
    # Process the input audio and get MFCC features
    mfcc_features = process_input(audio_blob, track_duration=30)

    # Initialize lists to store predicted genres and probabilities for each segment
    predicted_genres = []
    probabilities = []

    for mfcc_segment in mfcc_features:
        # Perform genre prediction using the loaded model
        X_to_predict = np.array(mfcc_segment)[np.newaxis, ..., np.newaxis]
        prediction = model.predict(X_to_predict)

        # Define genre labels
        genre_dict = {0: "Disco", 1: "Pop", 2: "Classical", 3: "Metal", 4: "Rock", 5: "Blues", 6: "Hiphop", 7: "Reggae", 8: "Country", 9: "Jazz"}

        # Get the predicted genre based on the highest probability
        predicted_index = np.argmax(prediction, axis=1)
        predicted_genre = genre_dict[int(predicted_index)]

        # Get the predicted probabilities for each genre
        probabilities.append(prediction.tolist()[0])

        # Store the predicted genre for this segment
        predicted_genres.append(predicted_genre)

    return predicted_genres[0], probabilities
# def record_audio(duration=30):
#     audio = pyaudio.PyAudio()

#     stream = audio.open(format=FORMAT, channels=CHANNELS,
#                         rate=RATE, input=True,
#                         frames_per_buffer=CHUNK)

#     frames = []
#     print("Recording...")
#     for i in range(0, int(RATE / CHUNK * duration)):
#         data = stream.read(CHUNK)
#         frames.append(data)
#     print("Finished recording")

#     stream.stop_stream()
#     stream.close()
#     audio.terminate()
#     frames_amplified = []
#     for frame in frames:
#         audio_array = np.frombuffer(frame, dtype=np.int16)
#         amplified_audio = audio_array * AMPLIFICATION_FACTOR
#         amplified_frame = amplified_audio.astype(np.int16).tobytes()
#         frames_amplified.append(amplified_frame)

#     #timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#     audio_filename = os.path.join(UPLOAD_FOLDER, f'recorded_audio.wav')

#     wf = wave.open(audio_filename, 'wb')
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(audio.get_sample_size(FORMAT))
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(frames))
#     wf.close()

#     return audio_filename


def visualize_prediction(predicted_genre, probabilities):
    genres = ["Disco", "Pop", "Classical", "Metal", "Rock", "Blues", "Hiphop", "Reggae", "Country", "Jazz"]
    mean_probabilities = np.mean(probabilities, axis=0)
    plt.figure(figsize=(8, 6))
    plt.bar(genres, mean_probabilities, color='royalblue')
    plt.xlabel('Genres')
    plt.ylabel('Probabilities')
    plt.title('Genre Prediction')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the visualization as an image
    plot_path = 'static/genre_prediction.png'
    plt.savefig(plot_path)

    return plot_path


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' in request.files:
        audio_file = request.files['file']
        print(type(audio_file))
        if audio_file:

            predicted_genre, probabilities = predict_genre(audio_file)
            plot_path = visualize_prediction(predicted_genre, probabilities)
            return render_template('result.html', prediction={"genre": predicted_genre, "plot_path": plot_path,'probabilities':probabilities})
    return 'No file provided.'



@app.route('/record', methods=['POST'])
def record():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Ensure the 'uploads' folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Save the uploaded audio file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio.wav')
        file.save(file_path)
    


        # Make genre prediction
        #prediction = predict_genre(file_path)

        return "audio record successfully",200

   
    # audio_file = request.files['audio']
    # if audio_file:
    #     audio_file.save('uploads/recorded_audio.wav')
    #     return 'Audio uploaded successfully', 200
    # else:
    #     return 'Failed to upload audio', 400
    


    
    

    

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'file' in request.files:
        audio_blob = request.files['file']
    if audio_blob:
        predicted_genre, probabilities = predict_genre(audio_blob)
        
        plot_path = visualize_prediction(predicted_genre, probabilities)
        
        return render_template('demo1.html', prediction={"genre": predicted_genre, "plot_path": plot_path, 'probabilities': probabilities})
    else:
        return 'No audio found.'


if __name__ == '__main__':
    app.run(debug=True)
