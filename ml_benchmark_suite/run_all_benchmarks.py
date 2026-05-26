import os
import sys
import subprocess


def run_isolated_script(script_path, working_dir):
    """Uruchamia skrypt jako całkowicie odizolowany proces systemowy, czyszcząc RAM po zakończeniu."""
    print(f"\n🚀 [Orchestrator] Uruchamianie procesu: {script_path}...")

    # Wywołujemy skrypt w jego własnym folderze roboczym
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=working_dir,
        capture_output=False,  # Logi lecą na żywo do konsoli
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Proces {script_path} zakończył się błędem (Kod: {result.returncode})")
        sys.exit(result.returncode)
    print(f"✅ Proces {script_path} ukończony pomyślnie.")


def main():
    print("🔥 ======================================================= 🔥")
    print("      URUCHAMIANIE ZIOLOWANEGO POTOKU TRENINGOWEGO ML / DL   ")
    print("🔥 ======================================================= 🔥")

    # Definiujemy ścieżki bezwzględne od lokalizacji tego skryptu
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Start Scikit-Learn w osobnym procesie
    run_isolated_script(
        script_path=os.path.join(base_dir, "scikit_learn", "train_sklearn.py"),
        working_dir=os.path.join(base_dir, "scikit_learn")
    )

    # 2. Start TensorFlow_Keras w osobnym procesie
    run_isolated_script(
        script_path=os.path.join(base_dir, "tensorflow_keras", "train_tf.py"),
        working_dir=os.path.join(base_dir, "tensorflow_keras")
    )

    # 3. Start PyTorch w osobnym procesie
    run_isolated_script(
        script_path=os.path.join(base_dir, "pytorch", "train_pytorch.py"),
        working_dir=os.path.join(base_dir, "pytorch")
    )

    print("\n🏁 [Orchestrator] Wszystkie odizolowane treningi ukończone pomyślnie!")

    # 4. Na samym końcu bezpiecznie wywołujemy skrypt tabeli zbiorczej
    print("\n📊 Generowanie tabeli końcowej turnieju...")
    run_isolated_script(
        script_path=os.path.join(base_dir, "run_benchmark.py"),
        working_dir=base_dir
    )


if __name__ == "__main__":
    main()
