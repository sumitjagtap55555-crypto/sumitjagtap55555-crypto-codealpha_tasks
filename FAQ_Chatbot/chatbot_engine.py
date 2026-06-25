"""
FAQ Chatbot Engine
------------------
Core NLP pipeline for a FAQ-matching chatbot:

  1. A small FAQ knowledge base (question/answer/category triples).
  2. Text preprocessing with NLTK (lowercasing, tokenization, stopword
     removal, lemmatization).
  3. Vectorization of all FAQ questions with TF-IDF.
  4. Matching an incoming user question against the FAQ questions using
     cosine similarity, and returning the best answer (or a graceful
     fallback if nothing matches well).

This file has no web dependency - run it directly for a terminal chat:

    python chatbot_engine.py

It's also imported by app.py, which wraps the same engine in a small
Flask web UI.
"""

import random
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# 0. Make sure the NLTK data this script needs is available locally.
#    (Safe to call every run - it skips anything already downloaded.)
# ---------------------------------------------------------------------
def _ensure_nltk_data():
    required = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, pkg in required.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_data()


# ---------------------------------------------------------------------
# 1. FAQ knowledge base
#    Swap this out (or load from a CSV/JSON file) for your own product.
# ---------------------------------------------------------------------
FAQ_DATA = [
    {
        "category": "Orders",
        "question": "How do I place an order?",
        "answer": "Browse to any product page, choose your options, and click "
                   "'Add to Cart'. When you're ready, open your cart and select "
                   "'Checkout' to enter shipping and payment details.",
    },
    {
        "category": "Orders",
        "question": "Can I cancel my order after placing it?",
        "answer": "You can cancel an order for free within 1 hour of placing it "
                   "from the 'My Orders' page. After that, the order may already "
                   "be in processing, so please contact support instead.",
    },
    {
        "category": "Orders",
        "question": "How can I track my order?",
        "answer": "Go to 'My Orders' and select the order you want to track. "
                   "You'll see live courier updates there, and we'll also email "
                   "you a tracking link once the order ships.",
    },
    {
        "category": "Orders",
        "question": "Can I change the shipping address after ordering?",
        "answer": "Yes, as long as the order hasn't shipped yet. Go to 'My "
                   "Orders', select the order, and choose 'Edit shipping "
                   "address'. Once it's shipped, the address can't be changed.",
    },
    {
        "category": "Shipping",
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 3-5 business days. Express shipping "
                   "(available at checkout for an extra fee) takes 1-2 business "
                   "days. Delivery estimates are shown before you pay.",
    },
    {
        "category": "Shipping",
        "question": "Do you ship internationally?",
        "answer": "We currently ship to over 40 countries. You'll see whether "
                   "your country is supported, along with shipping cost and "
                   "estimated delivery time, once you enter your address at "
                   "checkout.",
    },
    {
        "category": "Shipping",
        "question": "How much does shipping cost?",
        "answer": "Standard shipping is free on orders over $50, and $4.99 "
                   "otherwise. Express shipping costs $12.99 regardless of "
                   "order size.",
    },
    {
        "category": "Returns",
        "question": "What is your return policy?",
        "answer": "You can return most items within 30 days of delivery for a "
                   "full refund, as long as they're unused and in their original "
                   "packaging. Some categories (like opened electronics) may "
                   "have shorter windows - check the product page for details.",
    },
    {
        "category": "Returns",
        "question": "How do I start a return?",
        "answer": "Go to 'My Orders', select the item you want to return, and "
                   "click 'Start a return'. We'll email you a prepaid shipping "
                   "label - just box up the item and drop it off at any "
                   "supported carrier location.",
    },
    {
        "category": "Returns",
        "question": "How long does a refund take to process?",
        "answer": "Once we receive your returned item, refunds are processed "
                   "within 3-5 business days. It may take a few extra days for "
                   "your bank to show the credit on your statement.",
    },
    {
        "category": "Returns",
        "question": "Can I exchange an item instead of returning it?",
        "answer": "Yes. Choose 'Exchange' instead of 'Return' on the 'My "
                   "Orders' page, pick the new size, color, or model, and "
                   "we'll ship the replacement as soon as we receive your "
                   "original item.",
    },
    {
        "category": "Payments",
        "question": "What payment methods do you accept?",
        "answer": "We accept all major credit and debit cards, PayPal, Apple "
                   "Pay, Google Pay, and select buy-now-pay-later providers at "
                   "checkout.",
    },
    {
        "category": "Payments",
        "question": "Is it safe to use my credit card on this site?",
        "answer": "Yes. All payments are processed over an encrypted "
                   "connection, and we never store your full card number on "
                   "our servers - that's handled entirely by our PCI-compliant "
                   "payment provider.",
    },
    {
        "category": "Payments",
        "question": "Why was my payment declined?",
        "answer": "This is usually due to a typo in the card details, "
                   "insufficient funds, or your bank flagging the transaction "
                   "for extra verification. Double-check your details, or "
                   "contact your bank, then try again.",
    },
    {
        "category": "Payments",
        "question": "Can I get an invoice for my order?",
        "answer": "Yes - an itemized invoice is emailed to you automatically "
                   "after checkout, and you can also download it anytime from "
                   "'My Orders' by clicking 'View invoice'.",
    },
    {
        "category": "Account",
        "question": "How do I reset my password?",
        "answer": "Click 'Forgot password' on the sign-in page and enter your "
                   "email. We'll send a reset link that's valid for 30 minutes.",
    },
    {
        "category": "Account",
        "question": "How do I delete my account?",
        "answer": "Go to 'Account Settings' > 'Privacy' and select 'Delete my "
                   "account'. This permanently removes your data after a "
                   "14-day grace period, during which you can cancel the "
                   "request.",
    },
    {
        "category": "Account",
        "question": "How do I update my email address?",
        "answer": "Go to 'Account Settings' > 'Profile', update your email, "
                   "and confirm the change via the verification link we send "
                   "to your new address.",
    },
    {
        "category": "Warranty",
        "question": "Do your products come with a warranty?",
        "answer": "Most electronics come with a 1-year manufacturer warranty "
                   "covering defects. You can find the exact warranty length "
                   "on each product page under 'Specifications'.",
    },
    {
        "category": "Warranty",
        "question": "How do I make a warranty claim?",
        "answer": "Go to 'My Orders', select the item, and click 'Request "
                   "warranty service'. You'll need the order number and a "
                   "short description of the issue - our team typically "
                   "responds within 2 business days.",
    },
    {
        "category": "Support",
        "question": "How do I contact customer support?",
        "answer": "You can chat with us right here, email support@example.com, "
                   "or call +1-800-555-0199 between 9am-6pm (Mon-Fri).",
    },
    {
        "category": "Support",
        "question": "What are your customer support hours?",
        "answer": "Live chat and phone support are available 9am-6pm Monday "
                   "through Friday. Email is monitored 7 days a week, with "
                   "replies typically within 24 hours.",
    },
    {
        "category": "Promotions",
        "question": "Do you have a student discount?",
        "answer": "Yes - verified students get 10% off. Verify your student "
                   "status on the 'Student Discount' page to receive a unique "
                   "code.",
    },
    {
        "category": "Promotions",
        "question": "How do I use a discount code?",
        "answer": "Enter your code in the 'Promo code' box on the checkout "
                   "page and click 'Apply'. The discount will be reflected in "
                   "your order total before you pay.",
    },
    {
        "category": "Promotions",
        "question": "Can I combine multiple discount codes?",
        "answer": "Only one promo code can be applied per order, but it will "
                   "automatically stack with any site-wide sale that's "
                   "currently running.",
    },
]
# ---------------------------------------------------------------------
# 1b. Small talk - greetings, thanks, farewells, etc.
#
#     These are handled separately from the FAQ matcher on purpose.
#     Stuff like "hi" or "how are you" is almost entirely stopwords
#     ("how", "are", "you" all get stripped by the FAQ preprocessor),
#     so there's nothing left for TF-IDF/cosine similarity to work with.
#     A short list of regex patterns is a much more reliable way to
#     catch conversational chit-chat than trying to force it through
#     the same pipeline built for matching real FAQ questions.
# ---------------------------------------------------------------------
SMALL_TALK_INTENTS = [
    {
        "intent": "greeting",
        "patterns": [
            r"\b(hi+|hello+|hey+|hiya|heya|yo)\b",
            r"\bgood (morning|afternoon|evening)\b",
            r"\bgreetings\b",
        ],
        "responses": [
            "Hi! How are you doing today?",
            "Hello there! What can I help you with?",
            "Hey! How's it going? Let me know what you need help with.",
        ],
    },
    {
        "intent": "how_are_you",
        "patterns": [
            r"\bhow('?s| is| are) (it going|things|you doing|you)\b",
            r"\bwhat'?s up\b",
        ],
        "responses": [
            "I'm doing great, thanks for asking! What can I help you with?",
            "All good on my end! How can I assist you today?",
        ],
    },
    {
        "intent": "thanks",
        "patterns": [r"\bthanks?\b", r"\bthank you\b", r"\bappreciate it\b", r"\bthx\b"],
        "responses": [
            "You're welcome! Anything else I can help with?",
            "Happy to help! Let me know if you have more questions.",
        ],
    },
    {
        "intent": "farewell",
        "patterns": [r"\bbye\b", r"\bgoodbye\b", r"\bsee (you|ya)\b", r"\btalk to you later\b"],
        "responses": [
            "Goodbye! Have a great day.",
            "Bye! Feel free to come back anytime you have a question.",
        ],
    },
    {
        "intent": "bot_identity",
        "patterns": [r"\bwho are you\b", r"\bwhat are you\b", r"\bare you (a bot|human|real)\b"],
        "responses": [
            "I'm an FAQ assistant - I match your question against a knowledge "
            "base using TF-IDF and cosine similarity, then answer the ones "
            "I'm confident about.",
        ],
    },
]


class SmallTalkMatcher:
    """Lightweight rule-based matcher for conversational chit-chat."""

    def __init__(self, intents=SMALL_TALK_INTENTS):
        self.compiled = [
            (intent["responses"], [re.compile(p, re.IGNORECASE) for p in intent["patterns"]])
            for intent in intents
        ]

    def check(self, text: str):
        for responses, patterns in self.compiled:
            if any(p.search(text) for p in patterns):
                return random.choice(responses)
        return None


# ---------------------------------------------------------------------
class TextPreprocessor:
    """Lowercases, tokenizes, strips punctuation/stopwords, and lemmatizes."""

    def __init__(self):
        self._stopwords = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()
        self._punct_table = str.maketrans("", "", string.punctuation)

    def clean(self, text: str) -> str:
        text = text.lower().translate(self._punct_table)
        tokens = word_tokenize(text)
        tokens = [
            self._lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok.isalpha() and tok not in self._stopwords
        ]
        return " ".join(tokens)


# ---------------------------------------------------------------------
# 3. TF-IDF + cosine similarity matcher
# ---------------------------------------------------------------------
class FAQMatcher:
    """Matches a free-text user question against a fixed FAQ list."""

    def __init__(self, faq_data=FAQ_DATA, similarity_threshold: float = 0.25):
        self.faq_data = faq_data
        self.threshold = similarity_threshold
        self.preprocessor = TextPreprocessor()
        self.small_talk = SmallTalkMatcher()

        cleaned_questions = [
            self.preprocessor.clean(item["question"]) for item in self.faq_data
        ]
        self.vectorizer = TfidfVectorizer()
        self.faq_matrix = self.vectorizer.fit_transform(cleaned_questions)

    def best_match(self, user_question: str):
        """
        Returns a dict with the best-matching FAQ entry, the similarity
        score (0-1), and a flag telling you whether it cleared the
        confidence threshold. Greetings/thanks/farewells are caught by
        the small-talk layer first and short-circuit the FAQ lookup.
        """
        small_talk_reply = self.small_talk.check(user_question)
        if small_talk_reply is not None:
            return {
                "matched": True,
                "type": "smalltalk",
                "answer": small_talk_reply,
                "question": None,
                "score": 1.0,
            }

        cleaned = self.preprocessor.clean(user_question)
        if not cleaned.strip():
            return {
                "matched": False,
                "type": "fallback",
                "answer": "I didn't quite catch a question there - could you "
                          "rephrase that?",
                "question": None,
                "score": 0.0,
            }

        user_vector = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(user_vector, self.faq_matrix)[0]
        best_idx = similarities.argmax()
        best_score = round(float(similarities[best_idx]), 4)

        if best_score < self.threshold:
            return {
                "matched": False,
                "type": "fallback",
                "answer": "I'm not confident I have an answer for that. Try "
                          "rephrasing, or ask about orders, shipping, returns, "
                          "payments, your account, warranty, or support.",
                "question": None,
                "score": best_score,
            }

        match = self.faq_data[best_idx]
        return {
            "matched": True,
            "type": "faq",
            "answer": match["answer"],
            "question": match["question"],
            "category": match["category"],
            "score": best_score,
        }

    def top_matches(self, user_question: str, k: int = 3):
        """Returns the top-k FAQ candidates with their scores (for debugging/UI)."""
        cleaned = self.preprocessor.clean(user_question)
        user_vector = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(user_vector, self.faq_matrix)[0]
        ranked = sorted(
            range(len(similarities)), key=lambda i: similarities[i], reverse=True
        )[:k]
        return [
            {
                "question": self.faq_data[i]["question"],
                "score": round(float(similarities[i]), 4),
            }
            for i in ranked
        ]


# ---------------------------------------------------------------------
# 4. Simple terminal chat loop
# ---------------------------------------------------------------------
def run_cli_chat():
    matcher = FAQMatcher()
    print("FAQ Bot: Hi! Ask me about orders, shipping, returns, payments, your "
          "account, warranty, or support. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFAQ Bot: Bye!")
            break
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("FAQ Bot: Bye! Have a great day.")
            break
        result = matcher.best_match(user_input)
        print(f"FAQ Bot: {result['answer']}")
        if result.get("type") == "faq":
            print(f"         (matched: \"{result['question']}\" · "
                  f"confidence {result['score']:.2f})\n")
        else:
            print()


if __name__ == "__main__":
    run_cli_chat()
