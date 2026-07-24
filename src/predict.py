import joblib
import pandas as pd

model = joblib.load("C:\\Users\\newbl\\OneDrive\\Desktop\\Exam score predictor\\models\\exam_score_predictor.pkl")

hours_studied = float(input("Enter hours studied: "))
sleep_hours = float(input("Enter sleep hours: "))
attendance_percent = float(input("Enter attendance percentage: "))
previous_scores = float(input("Enter previous score: "))

new_student = pd.DataFrame({
    "hours_studied": [hours_studied],
    "sleep_hours": [sleep_hours],
    "attendance_percent": [attendance_percent],
    "previous_scores": [previous_scores]
})

prediction = model.predict(new_student)
print(f"\nPredicted Exam Score: {prediction[0]:.2f}")