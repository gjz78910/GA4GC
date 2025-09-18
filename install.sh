sudo apt install python3.10-venv
sudo apt install python3-pip
pip install --upgrade pip

# Core agent + SWE-Perf stack
pip install torch transformers datasets evaluate accelerate
pip install omegaconf hydra-core scikit-learn numpy pandas tqdm requests

# Optimization
pip install pymoo

# Optional (only if using these backends)
pip install mistralai   # for Mistral API
pip install ollama      # for local Ollama (still need Ollama server installed separately)
pip install django

cd SWE-Perf
pip install -r requirements.txt
