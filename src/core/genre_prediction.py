import joblib
import numpy as np
import pandas as pd
from typing import Dict
from numpy.typing import NDArray
import audio_feature_extractor as extractor

class MusicGenreClassifier:
    def __init__(self, model_path: str,
                 encoder_path: str,
                 sampling_rate: int = extractor.SAMPLING_RATE):
        """
        Initialize the music genre classifier.
        
        Parameters:
            model_path: Path to the trained XGBoost model.
            encoder_path: Path to the label encoder.
            sampling_rate: Sampling rate for audio processing.
        """
        self.sampling_rate = sampling_rate
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
    
    def classify_genre(self, audio_data: NDArray[np.float32]) -> str:
        """
        Classify the genre of an audio file based on its raw audio data.
        
        Parameters:
            audio_data: Raw audio data as a NumPy array.
        
        Returns:
            The predicted genre of the audio file.
        """
        features = extractor.extract_features(audio_data)
        features_df = pd.DataFrame([features])
        
        # Perform prediction
        y_pred_encoded = self.model.predict(features_df)
        print(f"y_pred_encoded: {y_pred_encoded}")
        
        # Decode the predicted label
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        print(f"y_pred: {y_pred}")
        
        return y_pred[0]
    
    def predict_probabilities(self, audio_data: NDArray[np.float32]) -> Dict[str, float]:
        """
        Get the probabilities that the audio file belongs to each genre.
        
        Parameters:
            audio_data: Raw audio data as a NumPy array.
        
        Returns:
            A dictionary with genres as keys and their corresponding
                probabilities as values.
        """
        features = extractor.extract_features(audio_data)
        features_df = pd.DataFrame([features])
        
        probabilities = self.model.predict_proba(features_df)[0]
        
        # Map probabilities to genre labels
        genre_probabilities = dict(zip(self.label_encoder.classes_, probabilities))

        return genre_probabilities
