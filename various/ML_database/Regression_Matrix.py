from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

#classification metrics
actual_labels =    [0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
predicted_labels = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1]

accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels)
recall = recall_score(actual_labels, predicted_labels)
f1_score = f1_score(actual_labels, predicted_labels)
confusion = confusion_matrix(actual_labels, predicted_labels)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1_score: {f1_score}")
print(f"Confusion: {confusion}")

#Regression Metrics
actual_values =    [10, 15, 20, 25, 30] #20
predicted_values = [12, 14, 18, 28, 33] #21

mae = mean_absolute_error(actual_values, predicted_values)
mse = mean_squared_error(actual_values, predicted_values)
r2s = r2_score(actual_values, predicted_values)
mdae = median_absolute_error(actual_values, predicted_values)

print(f"Mean absolute error: {mae}")
print(f"Mean squared error: {mse}")
print(f"R2 score: {r2s}")
print(f"Median absolute error: {mdae}")
