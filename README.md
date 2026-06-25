# FAQ Chatbot

A FAQ-matching chatbot: type a question, it preprocesses the text with
NLTK (tokenize, remove stopwords, lemmatize), vectorizes the FAQ list
with TF-IDF, and picks the best answer using cosine similarity.

## Files

- `chatbot_engine.py` — the actual NLP pipeline: FAQ data, preprocessing,
  TF-IDF + cosine similarity matching, plus a small-talk layer for
  greetings/thanks/farewells. Runs on its own as a terminal chat.
- `app.py` — a small Flask app that wraps the engine in a `/api/chat` endpoint.
- `templates/index.html` — the browser chat UI (talks to `/api/chat`).
- `requirements.txt` — Python dependencies.

## Small talk

Greetings ("hi", "hello", "good morning"), "how are you", "thanks",
"bye", and "who are you" are caught by a small set of regex patterns in
`SMALL_TALK_INTENTS` before the FAQ matcher ever runs, and answered with
a randomly-picked friendly reply. This exists because phrases like "how
are you" are almost entirely stopwords ("how", "are", "you" all get
stripped during FAQ preprocessing), so TF-IDF/cosine similarity has
nothing left to compare - a quick rule-based check is far more reliable
for this kind of short conversational text than forcing it through the
FAQ pipeline. Add more small-talk patterns/responses by editing
`SMALL_TALK_INTENTS` near the top of `chatbot_engine.py`.

## Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

## Option A — terminal chat (no web server)

```bash
python chatbot_engine.py
```

Type questions, type `quit` to exit.

## Option B — browser chat UI

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Click a category chip
or type your own question.

## Using your own FAQs

Edit the `FAQ_DATA` list at the top of `chatbot_engine.py` — each entry is
just `{"category": ..., "question": ..., "answer": ...}`. The matcher
re-fits itself automatically from whatever is in that list, so no other
code needs to change.

## Tuning match sensitivity

`FAQMatcher(similarity_threshold=0.25)` controls how confident a match has
to be before the bot will use it (0–1 scale). Lower it if the bot is being
too cautious and falling back to "I'm not confident…" too often; raise it
if it's matching unrelated questions.
