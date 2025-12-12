## Internal Agenda – CAIP 01 Thursday

Focus
- Help students move beyond “accuracy is everything”
- Show how metrics depend on use case (fraud, cancer detection, etc.)

Topics
- Evaluation Metrics:
  - Accuracy, Precision, Recall, F1
- Confusion Matrix
  - TP, TN, FP, FN — what they mean
- Business framing:
  - When is a false positive more costly than a false negative?
- Real-world examples:
  - Fraud detection, spam filters, medical diagnosis

Demo Plan
1. Load the model from Monday (or retrain it)
2. Use `classification_report` and `confusion_matrix`
3. Plot confusion matrix visually
4. Adjust threshold using predicted probabilities
5. Show how precision and recall change

Teaching Prompts
- “Would this metric still be useful if only 1% of cases are positive?”
- “What does a ‘false positive’ mean in fraud vs healthcare?”
- “How do you explain the model’s strengths and weaknesses to a business person?”
