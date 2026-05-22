import json
import glob
import sys


def verify_models():
    print("🛡️ [MLOps] Sprawdzanie kryteriów produkcyjnych...")
    json_files = glob.glob("benchmark_results/*.json")

    threshold = 0.83  # Wymagamy minimum 83% skuteczności
    failed = False

    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            if data["accuracy"] < threshold:
                print(
                    f"❌ MODEL ODRZUCONY: {data['framework']} - {data['model_name']} ma celność {data['accuracy']:.2%}, a wymagamy {threshold:.2%}")
                failed = True
            else:
                print(f"✅ MODEL ZATWIERDZONY: {data['framework']} - {data['model_name']} ({data['accuracy']:.2%})")

    if failed:
        print("🚨 [CI/CD Pipeline] Blokada wdrożenia z powodu niespełnienia norm jakościowych!")
        sys.exit(1)  # Zwrócenie kodu 1 przerywa automatyczny proces Airflow/GitHub Actions
    else:
        print("🚀 [CI/CD Pipeline] Wszystkie modele stabilne. Gotowe do wysłania na produkcję.")


if __name__ == "__main__":
    verify_models()
