from flask import Flask, render_template, request, jsonify
import joblib  # for loading your trained model
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize the stemmer
port_stem = PorterStemmer()

# Function to preprocess and stem the content
def preprocess(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

app = Flask(__name__)

# Load the trained model
clf_dt = joblib.load('model.pkl')  # Replace with your model file name
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    content = data['content']  # Extract content from the JSON request
    
    # Preprocess the content (stemming, TF-IDF vectorization, etc.) similar to your previous code
    processed_content = preprocess(content)
    vectorized_content = vectorizer.transform([processed_content])
    
    prediction = clf_dt.predict(vectorized_content)
    if prediction[0] == 'REAL':
        result = 'Real'
    else:
        result = 'Fake'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)

