import os
import re
from pathlib import Path

# Path to your chapters folder based on your previous terminal logs
CHAPTERS_DIR = Path(r"C:\Users\Javier\Documents\Junior_Seminar\junior-seminar-project-journal-and-research-report-chapters-javito350 - Copy\report")

# The flawless, academically-toned paragraph using your real 10-shot EVT data
NEW_PARAGRAPH = r"""Concrete Calibration Example: To illustrate this mechanism empirically, consider a real 10-shot calibration run on the `bottle` category (Seed: 111). The memory bank contains normal feature vectors. During calibration, a set of validation distances is generated. In this actual run, the sample drawn from the top 1% of calibration distances yielded $X_{tail} = \{21.139, 20.984, 20.777, 20.599\}$. The Generalized Pareto Distribution (GPD) fits these points, outputting a shape parameter of $\xi = +0.2357$ and a scale parameter of $\sigma = 1.9334$. Setting our target False Positive Rate at $10^{-2}$ (1%), the inverse CDF of the fitted GPD mathematically calculates the decision boundary. This calculation output a precise threshold of $\tau = 23.5764$. Any future inference distance $d(x) > 23.5764$ is thus mathematically classified as anomalous, grounding the theoretical EVT model in a tangible, data-driven decision boundary."""

def fix_fabricated_data():
    if not CHAPTERS_DIR.exists():
        print(f"[ERROR] Could not find chapters directory at: {CHAPTERS_DIR}")
        print("Please check the path and try again.")
        return

    # Regex to find the exact paragraph starting with "Concrete Calibration Example:" 
    # and ending with "tangible decision boundary."
    pattern = re.compile(r"Concrete Calibration Example:.*?tangible decision boundary\.", re.DOTALL)
    
    found = False

    # Search through all .qmd files in your report folder
    for filepath in CHAPTERS_DIR.glob("*.qmd"):
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()

        # Check if the fake numbers are in this file
        if "0.35, 0.38, 0.39, 0.41" in content:
            found = True
            print(f"[INFO] Found fabricated data in: {filepath.name}")

            # Swap the fake paragraph for the real one
            updated_content = pattern.sub(NEW_PARAGRAPH, content)

            with open(filepath, "w", encoding="utf-8") as file:
                file.write(updated_content)

            print(f"[SUCCESS] Perfectly updated {filepath.name} with real EVT data!")

    if not found:
        print("[INFO] Could not find the fabricated data. It may have already been replaced!")

if __name__ == "__main__":
    fix_fabricated_data()