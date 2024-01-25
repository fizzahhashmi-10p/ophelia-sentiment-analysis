import tensorflow as tf
import joblib
from transformers import BertTokenizer, TFBertForSequenceClassification


class ModelPredictor:
    def __init__(self, model_path='saved_model.pb'):
        self.model = TFBertForSequenceClassification.from_pretrained('model/', num_labels=3)  # Assuming 3 classes (good, neutral, bad)
        # model.summary()

        self.loaded_label_encoder = joblib.load('model/bert_sentiment_label_encoder.joblib')

        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('model/')

    def predict(self, data):
        tokens = self.tokenizer.encode(data,  return_tensors='tf', padding=True, truncation=True)
        max_len = len(tokens)

        padded_sequences_test = tf.keras.preprocessing.sequence.pad_sequences(tokens, padding='post', maxlen=max_len)

        predictions = self.model.predict(padded_sequences_test)
        predicted_labels_test = tf.argmax(predictions["logits"], axis=1).numpy()

        predicted_labels_decoded = self.loaded_label_encoder.inverse_transform(predicted_labels_test)
        print(predicted_labels_decoded)

        return predicted_labels_decoded[0]