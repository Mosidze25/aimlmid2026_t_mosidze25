import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("t_mosidze25_81367.csv")

X = df[["words", "links", "capital_words", "spam_word_count"]]
y = df["is_spam"]

# -----------------------------
# Train / Test Split (70/30)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Model Coefficients
# -----------------------------
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print("\nLogistic Regression Coefficients:")
print(coefficients)


# -----------------------------
# Validation
# -----------------------------
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy:", accuracy)

# Step 4
plt.figure()
ax = sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Legitimate", "Spam"],
    yticklabels=["Legitimate", "Spam"]
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix Heatmap")

# FIX: enable cursor coordinate display
ax.format_coord = lambda x, y: f"x={x:.2f}, y={y:.2f}"

plt.show()

# Step 5
counts = df["is_spam"].value_counts().sort_index()

plt.figure()
plt.bar(["Legitimate", "Spam"], counts.values)
plt.xlabel("Email Class")
plt.ylabel("Count")
plt.title("Class Distribution of Emails")
plt.show()


# Step 6
def classify_email(words, links, capital_words, spam_word_count):
    sample = pd.DataFrame([{
        "words": words,
        "links": links,
        "capital_words": capital_words,
        "spam_word_count": spam_word_count
    }])
    
    prediction = model.predict(sample)[0]
    return "Spam" if prediction == 1 else "Legitimate"

# Example
print(classify_email(120, 5, 20, 8))


# Step 7
import re
import string
import pandas as pd

# -----------------------------
# Spam keyword list
# -----------------------------
SPAM_WORDS = {
    "free", "win", "winner", "money", "cash", "prize",
    "offer", "click", "buy", "urgent", "limited"
}

# -----------------------------
# Feature Extraction Function
# -----------------------------
def extract_features(email_text):
    # Normalize text
    tokens = email_text.split()
    
    cleaned_tokens = [
        word.strip(string.punctuation) for word in tokens
    ]
    
    # Total words
    word_count = len(cleaned_tokens)
    
    # Link detection (http, https, www)
    link_count = len(re.findall(r"(http[s]?://|www\.)", email_text.lower()))
    
    # Capitalized words (ignore punctuation)
    capital_words = sum(
        1 for w in cleaned_tokens if w.isupper() and len(w) > 1
    )
    
    # Spam keyword count
    spam_word_count = sum(
        1 for w in cleaned_tokens if w.lower() in SPAM_WORDS
    )
    
    return {
        "words": word_count,
        "links": link_count,
        "capital_words": capital_words,
        "spam_word_count": spam_word_count
    }

# -----------------------------
# Email Classification Function
# -----------------------------
def classify_email_text(email_text):
    features = extract_features(email_text)
    sample = pd.DataFrame([features])
    
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]
    
    print("Extracted features:", features)
    print(f"Spam probability: {probability:.3f}")
    
    return "Spam" if prediction == 1 else "Legitimate"

# -----------------------------
# Example Spam Email
# -----------------------------
spam_email = """
FREE MONEY!!! CLICK NOW!!!
WIN CASH PRIZES TODAY!!!
Limited time offer!!!
Visit http://spam-offer-now.com
"""

print("Spam Email Prediction:", classify_email_text(spam_email))

# -----------------------------
# Example Legitimate Email
# -----------------------------
legit_email = """
Hello team,
Please find the meeting agenda attached.
Let me know if you have any questions.
"""

print("Legitimate Email Prediction:", classify_email_text(legit_email))
