"""
Flask web app for the FAQ chatbot.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request, jsonify
from chatbot_engine import FAQMatcher, FAQ_DATA

app = Flask(__name__)

# Build the matcher once at startup - it fits the TF-IDF vectorizer over
# the FAQ questions, so we don't want to redo that on every request.
matcher = FAQMatcher()


@app.route("/")
def index():
    categories = sorted({item["category"] for item in FAQ_DATA})
    # One example question per category, used as clickable suggestion chips.
    suggestions = []
    seen = set()
    for item in FAQ_DATA:
        if item["category"] not in seen:
            suggestions.append({"category": item["category"], "question": item["question"]})
            seen.add(item["category"])

    return render_template(
        "index.html",
        categories=categories,
        faq_count=len(FAQ_DATA),
        suggestions=suggestions,
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    result = matcher.best_match(user_message)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
