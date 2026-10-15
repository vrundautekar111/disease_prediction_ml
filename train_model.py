
from pathlib import Path
import json
import random

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SYMPTOMS = [
    "fever", "cough", "fatigue", "headache",
    "sore_throat", "runny_nose", "body_ache",
    "nausea", "vomiting", "diarrhea",
    "stomach_pain", "itchy_skin", "skin_rash",
    "sneezing", "breathlessness", "joint_pain",
    "chills", "loss_of_taste", "dizziness",
    "frequent_urination", "increased_thirst",
    "blurred_vision", "chest_discomfort", "anxiety"
]

# Synthetic examples for an educational project only.
PROFILES = {
    "Cold-like demo pattern": [
        "runny_nose", "sneezing", "sore_throat", "cough"
    ],
    "Flu-like demo pattern": [
        "fever", "chills", "body_ache", "fatigue", "headache"
    ],
    "Stomach-upset demo pattern": [
        "nausea", "vomiting", "diarrhea", "stomach_pain"
    ],
    "Skin-irritation demo pattern": [
        "itchy_skin", "skin_rash"
    ],
    "Allergy-like demo pattern": [
        "sneezing", "runny_nose", "itchy_skin"
    ],
    "Migraine-like demo pattern": [
        "headache", "nausea", "dizziness"
    ],
    "Joint-pain demo pattern": [
        "joint_pain", "fatigue", "body_ache"
    ],
    "High-blood-sugar warning demo pattern": [
        "frequent_urination", "increased_thirst",
        "blurred_vision", "fatigue"
    ],
    "Respiratory-symptom demo pattern": [
        "cough", "breathlessness",
        "chest_discomfort", "fatigue"
    ],
    "Stress/anxiety demo pattern": [
        "anxiety", "chest_discomfort", "dizziness"
    ]
}


def generate_dataset(samples_per_class=180):
    rng = random.Random(42)
    rows = []

    for label, core_symptoms in PROFILES.items():
        for _ in range(samples_per_class):
            row = {symptom: 0 for symptom in SYMPTOMS}

            for symptom in core_symptoms:
                row[symptom] = int(rng.random() < 0.82)

            for symptom in SYMPTOMS:
                if symptom not in core_symptoms:
                    row[symptom] = int(rng.random() < 0.055)

            if sum(row.values()) == 0:
                row[rng.choice(core_symptoms)] = 1

            row["label"] = label
            rows.append(row)

    rng.shuffle(rows)
    return pd.DataFrame(rows)


def main():
    df = generate_dataset()

    df.to_csv(
        DATA_DIR / "synthetic_symptom_data.csv",
        index=False
    )

    X = df[SYMPTOMS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Synthetic test accuracy:", round(accuracy, 3))
    print("Note: This is not a measure of medical accuracy.")

    joblib.dump(model, MODEL_DIR / "disease_model.joblib")

    with open(
        MODEL_DIR / "symptoms.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(SYMPTOMS, file, indent=2)

    print("Dataset, model, and symptom list saved successfully.")


if __name__ == "__main__":
    main()
