import pandas as pd

data = {
    "Student Name": ["Amit", "Riya", "Sourav", "Neha", "Rahul"],
    "Marks": [72, 85, 68, 91, 77]
}
df = pd.DataFrame(data)

def assign_grade(mark):
    if mark >= 90:
        return 'A'
    elif mark >= 80:
        return 'B'
    elif mark >= 70:
        return 'C'
    else:
        return 'D'

df["Grade"] = df["Marks"].apply(assign_grade)
print(df)
