import pandas as pd

data = pd.read_csv('data.csv')

X = data[[
        "StudyTimeWeekly",
        "Absences",
        "Age",
        "Tutoring",
        "Sports",
        "Extracurricular",
        "ParentalSupport",
        "Music"
    ]].values

print(X.shape[1])