import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset
data = {
    'text': [
        'Congratulations! You have won $1000. Click here now.',
        'Your account has been suspended. Verify immediately.',
        'Claim your free gift card now.',
        'Update your bank details to avoid account closure.',
        'Meeting scheduled for tomorrow at 10 AM.',
        'Please find the attached project report.',
        'Lunch meeting at 1 PM today.',
        'Project submission deadline extended.'
    ],
    'label': [
        'phishing',
        'phishing',
        'phishing',
        'phishing',
        'safe',
        'safe',
        'safe',
        'safe'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and labels
X = df['text']
y = df['label']

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['phishing', 'safe'])

print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=['Phishing', 'Safe'],
    yticklabels=['Phishing', 'Safe']
)

plt.title('Phishing Email Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Test custom email
test_email = [
    "Urgent! Click this link and verify your bank account immediately."
]

test_features = vectorizer.transform(test_email)
prediction = model.predict(test_features)

print("\nTest Email:")
print(test_email[0])
print("Prediction:", prediction[0])