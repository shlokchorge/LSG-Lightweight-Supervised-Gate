import os
os.environ["LSG_RESULTS_DIR"] = "results2"

from lsg.train import run, run_adaptation_curve, run_rigorous
from lsg.analysis import run_analysis

if __name__ == "__main__":
    os.makedirs("results2", exist_ok=True)
    run()
    run_adaptation_curve()
    run_rigorous()
    run_analysis()
