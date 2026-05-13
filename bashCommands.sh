# Create a fresh virtual environment
python -m venv ff_project

# Activate it
source ff_project/bin/activate # Mac/Linux

# Install dependencies if not already done so
# pip install yfinance pandas numpy matplotlib seaborn pyarrow requests lxml statsmodels

#run load_data.py if data file doesn't have the csv and parquet files   
python load_data.py
python models.py
python diagnostics.py
python alpha_analysis.py


#Deactivate
deactivate 