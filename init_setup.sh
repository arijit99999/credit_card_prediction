echo[$(date)]:'START'

echo[$(date)]:'creating env with python 3.11 version'
conda create --prefix ./env python=3.11 -y

echo[$(date)]:'activating the environment'
sorce activate ./myenv

echo[$(date)]:'installing the env requrements'
pip install -r requirements.txt

echo[$(date)]:'END'