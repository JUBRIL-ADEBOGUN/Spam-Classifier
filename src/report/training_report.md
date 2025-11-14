# Model Training Report

**Timestamp:** `2025-11-13 15:31:27.201317`

## 1. Performance Metrics

```
              precision    recall  f1-score   support

         HAM       0.88      1.00      0.94       966
        SPAM       1.00      0.11      0.19       149

    accuracy                           0.88      1115
   macro avg       0.94      0.55      0.56      1115
weighted avg       0.90      0.88      0.84      1115

```

## 2. Confusion Matrix

![Confusion Matrix](spam-classifier/src/report/confusion_matrix.png)

## 3. Feature Importance

This plot shows the top words that push the prediction towards SPAM (red) or HAM (green).

![Feature Importance](spam-classifier/src/report/feature_importance.png)
