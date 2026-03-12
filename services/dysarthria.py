import os
import pandas as pd
import librosa
import torch
from models.gru import BiGRUAudioClassifier

def dysarthria_classification(sound_file, processor_small, model_small):
    # MAPPING
    df = pd.read_excel("mapping.xlsx")
    # Filter to only the 155 classes the model was trained on (non-B prefix classes)
    df_filtered = df[~df['FILE NAME'].str.startswith('B')]
    mapping = df_filtered.set_index('FILE NAME').T.to_dict('list')
    classes = sorted(df_filtered['FILE NAME'].unique())
    class_to_idx = {i: c for i, c in enumerate(classes)}

    sample, sr = librosa.load(sound_file, sr=16000)
    input_feature = processor_small(sample, sampling_rate=sr, return_tensors="pt").input_features
    
    with torch.no_grad():
        images = model_small(input_feature).last_hidden_state
        
    saved_model = BiGRUAudioClassifier(768, 155, 256, 2)
    saved_model.load_state_dict(torch.load('models/weights/dys_model', map_location=torch.device('cpu')))
    saved_model.eval()
    
    images = images[-1, 0:500, :]
    images = images.unsqueeze(0).float()
    
    outputs = saved_model(images)
    _, prediction = torch.max(outputs.data, 1)
    
    file_name = class_to_idx[prediction.numpy()[0]]
    label = mapping[file_name][0]
    return label.replace('\\', '/'), label
