import io, zipfile, requests
import yfinance as yf
import pandas as pd
import numpy as np
from requests import Session
from io import StringIO
from pathlib import Path
import time

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
START, END = "2010-01-01", "2024-12-31"
WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"


#S&P 500 tickers
print("Fetching list of S&P 500 tickers from Wikipedia")
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(WIKI_URL, headers=headers)
response.raise_for_status()
sp500 = pd.read_html(StringIO(response.text))[0]
tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()
print(f"Found {len(tickers)} tickers")

#Monthly stcok prices via yfinance
print("\nDownloading monthly adjusted close prices")
raw = yf.download(
    tickers,
    start=START,
    end=END,
    interval="1mo",
    auto_adjust=True,
    progress=False,
)["Close"]

#Compute the log returns
log_ret = np.log(raw / raw.shift(1)).dropna(how="all")
log_ret.index = pd.to_datetime(log_ret.index).to_period("M").to_timestamp()

#save 
log_ret.to_csv(DATA_DIR/"returns.csv")
print("Saved returns.csv")
print(f"Factors shape: {log_ret.shape}, columns: {list(log_ret.columns)}")

#Fama-French 5-Factor data
def fetch_csv(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    fname = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    raw_text = z.read(fname).decode("utf-8", errors="replace")
    lines = raw_text.splitlines()
    # Find first line that starts with a 6-digit year-month integer
    data_start = next(i for i, l in enumerate(lines) if l.strip()[:6].isdigit())
    data_lines = [l for l in lines[data_start:] if l.strip()[:6].isdigit()]
    return pd.read_csv(io.StringIO("\n".join(data_lines)), header=None)

print("Downloading Fama-French 5-Factor data")
ff_raw = fetch_csv(FF_URL)
ff_raw.columns = ["yyyymm", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
ff_raw["date"] = pd.to_datetime(ff_raw["yyyymm"].astype(str), format="%Y%m")
ff = ff_raw.set_index("date")[["Mkt-RF","SMB","HML","RMW","CMA","RF"]].astype(float) / 100

print("Downloading Momentum factor")

mom_raw = fetch_csv(MOM_URL)
mom_raw.columns = ["yyyymm", "Mom"]
mom_raw["date"] = pd.to_datetime(mom_raw["yyyymm"].astype(str), format="%Y%m")
mom = mom_raw.set_index("date")[["Mom"]].astype(float) / 100

factors = ff.join(mom, how="inner").loc[START:END]
# Save
factors.to_csv(DATA_DIR / "factors.csv")
print(f"Saved factors.csv")
print(f"Factors shape: {factors.shape}, columns: {list(factors.columns)}")

#Build long-format panel
print("\nBuilding long-format panel")
panel = (
    log_ret.stack().reset_index()
    .rename(columns={"Date":"date","Ticker":"ticker",0:"ret"})
)
panel = panel.merge(factors.reset_index(), on="date", how="inner")
panel["excess_ret"] = panel["ret"] - panel["RF"]
panel = panel.dropna()

panel.to_parquet(DATA_DIR / "panel.parquet", index=False)
print(f"Saved panel.parquet  {panel.shape}")


# yahoo finance api is unreliable and doesn't allow 
# accessing more than a certain number of ticker in one request
# so the following code helps to retry fetching the failed tickers individually
failed = [t for t in tickers if t not in raw.columns or raw[t].isna().all()]
print(f"Retrying {len(failed)} failed tickers individually")
frames = [raw]
for ticker in failed:
    try:
        df = yf.Ticker(ticker).history(
            start=START, end=END,
            interval="1mo",
            auto_adjust=True
        )["Close"]
        if not df.empty:
            df.name = ticker
            frames.append(df)
    except Exception as e:
        print(f"  Permanently failed {ticker}: {e}")
    time.sleep(0.3)  #avoid rate limiting

raw = pd.concat(frames, axis=1)
raw = raw.loc[:, ~raw.columns.duplicated()]  #drop any duplicate columns

print("\nAll data saved to /data")