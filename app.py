# app.py

from flask import Flask, render_template, request, jsonify
from predict import ModelPredictor

app = Flask(__name__)
predictor = ModelPredictor()  # You can provide a specific model path if needed

@app.route('/', methods=['GET', 'POST'])
def index():
    
    default_sentiment = 'neutral'
    sentiment = default_sentiment
    user_input = 'What are you thinking?'
    

    if request.method == 'POST':
        try:
            # Assume the form has an input field with the name 'user_input'
            user_input = request.form.get('user_input')
            sentiment = predictor.predict([user_input])
        except Exception as e:
            return jsonify({'error': str(e)})
    print(sentiment)
    print(user_input)
    return render_template('index.html', sentiment=sentiment, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
