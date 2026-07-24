import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("C:\\Users\\newbl\\OneDrive\\Desktop\\Exam score predictor\\data\\student_exam_scores.csv")

X = df[[
    "hours_studied",
    "sleep_hours",
    "attendance_percent",
    "previous_scores"
]]

y = df["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("===== Model Performance =====")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

joblib.dump(model, "C:\\Users\\newbl\\OneDrive\\Desktop\\Exam score predictor\\models\\exam_score_predictor.pkl")
print("\nModel saved successfully!")