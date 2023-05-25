# aieng-template
aieng template repo, the static code checker runs on python3.8

# Environment Setup and Installing dependencies
```
module load python/3.9.10
python -m venv ad_anv
source ad_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

# using pre-commit hooks
To check your code at commit time
```
pre-commit install
```

You can also get pre-commit to fix your code
```
pre-commit run
```
