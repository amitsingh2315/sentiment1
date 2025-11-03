# Import the Flask app from api.py
from api import app

if __name__ == "__main__":
    app.run(port=5000, debug=True)

