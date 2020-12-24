from sklearn.ensemble import RandomForestClassifier
import models

# Fit a Basic Random Forest Model on these vectors

rf = RandomForestClassifier()
rf_model = rf.fit(models.X_train_vectors, models.y_train.values.ravel())

# Use the trained model to make predictions
y_pred = rf_model.predict(models.X_test_vectors)
print(y_pred)
