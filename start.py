"""
start.py — Lance tous les services ATS Intelligent en un seul script.
Auteur : Mourad Amoussa | RNCP39586

Usage :
    python start.py
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(label: str, cmd: list[str], shell_win: bool = False) -> None:
    """Lance une commande dans un terminal Windows séparé."""
    if sys.platform == "win32":
        subprocess.Popen(
            f'start "{label}" cmd /k "{" ".join(cmd)}"',
            shell=True,
            cwd=ROOT,
        )
    else:
        subprocess.Popen(cmd, cwd=ROOT)


print("=" * 60)
print("  🚀  ATS Intelligent — Démarrage des services")
print("=" * 60)

# 1. PostgreSQL via Docker Compose
print("\n[1/4] PostgreSQL (Docker)…")
subprocess.run(["docker", "compose", "up", "-d"], cwd=ROOT, check=False)
time.sleep(3)
print("      ✅ PostgreSQL sur le port 15432")

# 2. MLflow UI
print("\n[2/4] MLflow UI…")
run(
    "MLflow",
    [PY, "-m", "mlflow", "ui",
     "--backend-store-uri", "sqlite:///mlflow.db",
     "--port", "5001"],
)
time.sleep(2)
print("      ✅ http://localhost:5001")

# 3. FastAPI scoring
print("\n[3/4] FastAPI (scoring API)…")
run(
    "FastAPI",
    [PY, "-m", "uvicorn", "api.scoring_api:app",
     "--port", "8001", "--reload"],
)
time.sleep(2)
print("      ✅ http://localhost:8001/docs")

# 4. Streamlit app
print("\n[4/4] Streamlit…")
run(
    "Streamlit",
    [PY, "-m", "streamlit", "run", "app.py"],
)

print("\n" + "=" * 60)
print("  ✅  Tous les services démarrés !")
print()
print("  Streamlit   → http://localhost:8501")
print("  FastAPI     → http://localhost:8001/docs")
print("  MLflow      → http://localhost:5001")
print("  PostgreSQL  → localhost:15432 (ats_db)")
print("=" * 60)
