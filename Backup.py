import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import ttk
import numpy as np
import webbrowser

# Load the dataset
data = pd.read_csv('dataset.csv')

# Print dataset columns for debugging
print("Dataset columns:", data.columns.tolist())

# Select only relevant columns based on actual dataset
selected_columns = ['Average Academic score(%)', 'Hours working per day', 'hackathons', 'coding skills rating',
                    'can work long time before system?', 'self-learning capability?', 'Extra-courses did',
                    'certifications', 'workshops', 'talenttests taken?', 'Interested subjects',
                    'interested career area ', 'Job/Higher Studies?', 'Type of company want to settle in?',
                    'Taken inputs from seniors or elders', 'interested in games', 'Interested Type of Books',
                    'Management or Technical', 'hard/smart worker', 'worked in teams ever?', 'Suggested Job Role']

# Ensure only existing columns are selected
existing_columns = [col for col in selected_columns if col in data.columns]
data = data[existing_columns].dropna()

# Debug: Check row count after dropna
print("Rows after dropping missing values:", len(data))
if data.empty:
    raise ValueError("Dataset is empty after removing missing values. Ensure 'copy.csv' has valid data.")

# Encode categorical variables
label_encoders = {}
categorical_columns = ['can work long time before system?', 'self-learning capability?', 'Extra-courses did',
                       'certifications', 'workshops', 'talenttests taken?', 'Interested subjects',
                       'interested career area ', 'Job/Higher Studies?', 'Type of company want to settle in?',
                       'Taken inputs from seniors or elders', 'interested in games', 'Interested Type of Books',
                       'Management or Technical', 'hard/smart worker', 'worked in teams ever?']

# Ensure only existing categorical columns are encoded
categorical_columns = [col for col in categorical_columns if col in data.columns]
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Encode target variable
if 'Suggested Job Role' in data.columns:
    target_le = LabelEncoder()
    data['Suggested Job Role'] = target_le.fit_transform(data['Suggested Job Role'])
else:
    raise ValueError("'Suggested Job Role' column is missing from the dataset. Ensure it is present in 'copy.csv'.")

# Prepare dataset for model training
X = data.drop(columns=['Suggested Job Role'])
y = data['Suggested Job Role']

# Debug: Check if X and y are empty
if X.empty or y.empty:
    raise ValueError("Feature set (X) or target variable (y) is empty. Check the dataset for missing values or incorrect formatting.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_job_role():
    user_inputs = []
    for entry in entries:
        if entry['type'] == 'combo':
            encoded_value = label_encoders[entry['label']].transform([entry['widget'].get()])[0]
            user_inputs.append(encoded_value)
        else:
            try:
                user_inputs.append(float(entry['widget'].get()))
            except ValueError:
                result_label.config(text="Invalid input. Please enter numeric values correctly.")
                return
    
    user_inputs = np.array(user_inputs).reshape(1, -1)
    prediction = model.predict(user_inputs)
    predicted_role = target_le.inverse_transform(prediction)[0]
    
    result_label.config(text=f"Suggested Job Role: {predicted_role}")
    
    # Button to search job role on Google
    search_button.config(command=lambda: webbrowser.open(f"https://www.google.com/search?q={predicted_role} jobs"))
    search_button.grid(row=len(entries)//2 + 4, columnspan=2, pady=10)

def clear_inputs():
    for entry in entries:
        if entry['type'] == 'combo':
            entry['widget'].set('')
        else:
            entry['widget'].delete(0, tk.END)

# Initialize UI
root = tk.Tk()
root.title("Job Recommender System")
root.attributes('-fullscreen', True)

# Create a frame for the form
form_frame = ttk.Frame(root, padding="10")
form_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

canvas = tk.Canvas(form_frame)
scrollbar = ttk.Scrollbar(form_frame, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

entries = []
for idx, label in enumerate(X.columns):
    frame = ttk.Frame(scrollable_frame)
    frame.grid(row=idx//2, column=idx%2, padx=10, pady=5, sticky=tk.W+tk.E)
    
    lbl = ttk.Label(frame, text=label, background='#f0f0f0')
    lbl.pack(side=tk.LEFT, padx=5)
    
    if label in label_encoders:
        combo = ttk.Combobox(frame, values=list(label_encoders[label].classes_), background='#e6f7ff')
        combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        entries.append({'label': label, 'widget': combo, 'type': 'combo'})
    else:
        entry = ttk.Entry(frame, background='#e6f7ff')
        entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        entries.append({'label': label, 'widget': entry, 'type': 'entry'})

predict_button = ttk.Button(scrollable_frame, text="Predict Job Role", command=predict_job_role)
predict_button.grid(row=len(entries)//2 + 1, columnspan=2, pady=10)

clear_button = ttk.Button(scrollable_frame, text="Clear", command=clear_inputs)
clear_button.grid(row=len(entries)//2 + 2, columnspan=2, pady=10)

result_label = ttk.Label(scrollable_frame, text="", font=('Helvetica', 12, 'bold'))
result_label.grid(row=len(entries)//2 + 3, columnspan=2, pady=10)

search_button = ttk.Button(scrollable_frame, text="Search Jobs")
search_button.grid(row=len(entries)//2 + 4, columnspan=2, pady=10)

quit_button = ttk.Button(root, text="❌", command=root.quit, style="Quit.TButton")
quit_button.place(relx=1.0, rely=0.0, anchor='ne')

style = ttk.Style()
style.configure("Quit.TButton", background="Red", foreground="Red", font=('Helvetica', 12, 'bold'))

root.mainloop()
