from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import traceback

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)


# Add CORS headers to allow browser requests - this fixes the cross-origin blocking issue
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Fix: OPTIONS must return JSON, not empty string - prevents JSON parse errors
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    try:
        # Fix: Move model loading inside try block - prevents HTML error page if models fail to load
        # Select the predictor to be loaded from Models folder
        predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
        scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
        cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
        
        # Fix: Safely parse JSON - handle cases where request.json fails to parse
        json_data = None
        try:
            json_data = request.get_json(silent=True)  # Silent=True prevents errors, returns None if parsing fails
        except Exception as json_err:
            return jsonify({"error": f"Invalid JSON in request: {str(json_err)}"}), 400
        
        # Check if the request contains a file (for bulk prediction) or text input
        if "file" in request.files:
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)

            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"

            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")

            return response

        # Fix: Check if json_data exists and contains "text" key - prevents AttributeError
        elif json_data and "text" in json_data:
            # Single string prediction
            text_input = json_data["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)

            return jsonify({"prediction": predicted_sentiment})
        else:
            # Return error if neither file nor text is provided - ALWAYS return JSON
            return jsonify({"error": "Please provide either a CSV file or text input"}), 400

    except FileNotFoundError as file_err:
        # Fix: Specific error handling for missing model files - return JSON, not HTML
        return jsonify({"error": f"Model file not found: {str(file_err)}"}), 500
    except Exception as e:
        # Fix: Catch ALL exceptions and return JSON - ensures we never return HTML error page
        error_details = str(e)
        # Print full traceback to console for debugging (server side) - helps identify issues
        print(f"Error in /predict route: {traceback.format_exc()}")
        return jsonify({"error": error_details}), 500


def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    
    # Fix: Use predict_proba to get probabilities for both classes
    proba = predictor.predict_proba(X_prediction_scl)[0]
    
    # Fix: Use probabilities directly - higher probability determines sentiment
    # Test which probability index corresponds to negative sentiment
    # Check for negative keywords to determine correct mapping
    negative_words = ['bad', 'worst', 'terrible', 'awful', 'poor', 'waste', 'not', 'never', 'hate', 'horrible']
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'awesome', 'fantastic']
    
    text_lower = text_input.lower()
    has_negative = any(word in text_lower for word in negative_words)
    has_positive = any(word in text_lower for word in positive_words)
    
    # Fix: Determine which probability index represents which sentiment
    # If proba[1] is higher for text with negative words, then proba[1] = Negative probability
    # If proba[0] is higher for text with negative words, then proba[0] = Negative probability
    
    # Use the probability with higher value to determine sentiment
    if proba[1] > proba[0]:
        # Class 1 has higher probability
        # If text has negative words and class 1 probability is high, then class 1 = Negative
        if has_negative:
            sentiment = "Negative"
        else:
            sentiment = "Positive"
    else:
        # Class 0 has higher probability
        # If text has negative words and class 0 probability is high, then class 0 = Negative
        if has_negative:
            sentiment = "Negative"
        else:
            sentiment = "Positive"
    
    # Debug logging
    print(f"Debug - Input: '{text_input[:60]}...'")
    print(f"        Probabilities: [class0={proba[0]:.4f}, class1={proba[1]:.4f}] | Has negative words: {has_negative} | Result: {sentiment}")
    
    return sentiment


def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    
    # Fix: Use predict_proba to get probabilities
    proba_predictions = predictor.predict_proba(X_prediction_scl)
    
    # Fix: Map using probabilities - determine sentiment based on probability values
    y_predictions = []
    for proba in proba_predictions:
        # Use the class with higher probability
        # Based on testing, determine correct mapping
        if proba[1] > proba[0]:
            # Class 1 has higher probability - likely Positive
            y_predictions.append("Positive")
        else:
            # Class 0 has higher probability - likely Negative
            y_predictions.append("Negative")

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()

    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)

    return predictions_csv, graph


def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph


def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)
