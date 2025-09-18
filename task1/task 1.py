import pandas as pd
import numpy as np

df = pd.read_csv("Dataset 1.csv")
print("First 10 rows:")
print(df.head(10))

df['Total'] = df[['Math', 'Science', 'English']].sum(axis=1)

# Top 5 students by total marks
print("\nTop 5 students by total marks:")
top_5 = df.sort_values(by='Total', ascending=False).head(5)
print(top_5)

# Tuple Practice
# Tuple of (StudentID, Name) for first 5 students
student_tuples = tuple(zip(df['StudentID'][:5], df['Name'][:5]))
print("\nTuples of first 5 students:")
print(student_tuples)

#List Practice
# List of lists for marks of top 5 students
marks_list = top_5[['Math', 'Science', 'English']].values.tolist()

# Calculate and print average score
all_scores = [score for student in marks_list for score in student]
average_score = sum(all_scores) / len(all_scores)
print("\nAverage score of top 5 students:", round(average_score, 2))

#Set Practice
genders = set(df['Gender'].dropna())
print("\nUnique genders in dataset:", genders)

#Dictionary Practice
# Pick one student (first student)
student = df.iloc[0]
student_dict = {
    "Name": student['Name'],
    "Gender": student['Gender'],
    "Marks": {
        "Math": student['Math'],
        "Science": student['Science'],
        "English": student['English']
    }
}
print("\nStudent Dictionary:")
print(student_dict)

#NumPy Practice
scores = df[['Math', 'Science', 'English']].to_numpy()

# Mean, Max, Std Dev
print("\nMean per subject:", np.mean(scores, axis=0))
print("Max per subject:", np.max(scores, axis=0))
print("Std Dev per subject:", np.std(scores, axis=0))

# Filtering students with Math > 85 and Science > 90
filtered_df = df[(df['Math'] > 85) & (df['Science'] > 90)]
print("\nStudents with Math > 85 and Science > 90:")
print(filtered_df)

# Count of male and female students scoring above 240 total marks
high_scorers = df[df['Total'] > 240]
gender_counts = high_scorers['Gender'].value_counts()
print("\nHigh scorers (>240 marks) by gender:")
print(gender_counts)
