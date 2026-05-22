import os
import json
import glob
import pandas as pd


def compile_and_display_benchmark():
    print("📋 [ML-Suite] Szukanie plików wynikowych w benchmark_results/...")

    # Pobieramy ścieżki do wszystkich plików JSON w folderze wyników
    json_files = glob.glob("benchmark_results/*.json")

    if not json_files:
        print("⚠️ Brak plików JSON! Upewnij się, że uruchomiłeś skrypty treningowe.")
        return

    all_results = []

    # Wczytujemy każdy plik i dodajemy do wspólnej listy
    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            all_results.append(data)

    # Konwersja na elegancką tabelę Pandas
    df_results = pd.DataFrame(all_results)

    # Formatowanie wartości dla lepszej czytelności w konsoli
    df_results["accuracy"] = df_results["accuracy"].apply(lambda x: f"{x:.2%}")
    df_results["precision"] = df_results["precision"].apply(lambda x: f"{x:.2%}")
    df_results["recall"] = df_results["recall"].apply(lambda x: f"{x:.2%}")
    df_results["training_time_seconds"] = df_results["training_time_seconds"].apply(lambda x: f"{x:.2f}s")

    # Sortujemy ranking od najwyższej dokładności (Accuracy)
    df_results = df_results.sort_values(by="accuracy", ascending=False)

    print("\n🏆 ========================================================================= 🏆")
    print("                       WIELKI TURNIEJ FRAMEWORKÓW ML / DL                      ")
    print("🏆 ========================================================================= 🏆")
    print(df_results.to_string(index=False))
    print("===============================================================================\n")


if __name__ == "__main__":
    compile_and_display_benchmark()
