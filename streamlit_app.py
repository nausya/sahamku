import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import yfinance as yf
import numpy as np
import plotly.figure_factory as ff
from matplotlib import markers
from streamlit_option_menu import option_menu
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.sidebar.info('SELAMAT DATANG (Versi Beta)')
st.title('ANALITIK SAHAM INDONESIA') 

def main():
    selected2 = option_menu(None, ["Home", "Cari Data", "Screener", 'Prediksi'], icons=['house', 'file-earmark-text', 'sliders2-vertical', 'graph-up-arrow'], menu_icon="cast", default_index=0, orientation="horizontal")
    if selected2 == 'Cari Data':
         dataframe()
    elif selected2 == 'Screener':
         screener()
    elif selected2 == 'Prediksi':
         predict()
    else:
         tech_indicators()


#Halaman Utama
@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

#AMBIL KODE EMITEN DARI CSV

## Load the data
dataemiten = pd.read_csv('kodesaham.csv').sort_values('Kode')
 
## Get the list of countries
emiten = dataemiten['Kode'] + ' | ' + dataemiten['Nama Perusahaan']

 
## Create the select box
selected_emiten = st.sidebar.selectbox('Pilih Emiten:', emiten)

 
## Display the filtered data
st.header(selected_emiten.split(' | ')[1])

option = selected_emiten.split(' | ')[0] + ".JK"


#ganti value None
def ceknon(x):
    if x is not None:
       x = round(x,2)*100
       x = int(x)
       return x
    elif x == 0:
       return 0
    else:
       st.error('This is an error', icon="🚨")

#Display Persentil
saham = [option]
screensaham = []
for stock in saham:
        info = yf.Ticker(stock).info
        kode = stock.replace('.JK','')
        skg = info.get('currentPrice')
        lo  = info.get('fiftyTwoWeekLow')
        hi  = info.get('fiftyTwoWeekHigh')
        om  = info.get('operatingMargins')
        dev = info.get('payoutRatio')
        roe =  info.get('returnOnEquity')
        screensaham.append({'kode':kode,'skg':skg,'lo':lo,'hi':hi,'om':om,'dev':dev,'roe':roe})
df = pd.DataFrame(screensaham)
df = df.fillna(0)

L52 = df['lo']
H52 = df['hi']
C = df['skg']
D = (H52-L52)/100
try:
    P = (C - L52)/D 
except ZeroDivisionError:
    P = 0
st.subheader(f"Harga terkini Rp {int(C)} berada pada level {int(P)} dari skala 100", divider="rainbow")
om  = df['om']
dev = df['dev']
roe = df['roe']

st.subheader(f"MarginOps : {ceknon(om)}%, DevPR : {ceknon(dev)}%, ROE : {ceknon(roe)}%", divider="rainbow")
#Notasi Saham
kode = selected_emiten.split(' | ')[0]
n = pd.read_csv('notasi.csv')
n = n[(n['Kode'] == kode)]
if n.isna().empty:
   st.text("")
else:  
   n = n['Keterangan Notasi'].str.split('|', expand=False)
   for y in n:
    for x in y:
     st.error(x)

#st.markdown('notasi(kode)') # see *
st.info('Untuk jangka panjang perlu diperhatikan kisaran level harga kurang dari 10')


#Proses sidebar data
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Durasi Hari', value=1000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Tanggal Awal', value=before)
end_date = st.sidebar.date_input('Tanggal Akhir', today)
if st.sidebar.button('Proses'):
    if start_date < end_date:
        st.sidebar.success('Tanggal Awal: `%s`\n\nTanggal Akhir: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Terdapat Kesalahan: Tanggal akhir harus ditulis setelah tanggal awal')

data = download_data(option, start_date, end_date)
scaler = StandardScaler()

def tech_indicators():
    st.header('Teknikal Indikator')
    option = st.radio('Pilih Teknikal Indikator', ['Close', 'BB', 'MACD', 'RSI', 'EMA'])

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    #sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    #elif option == 'SMA':
        #st.write('Simple Moving Average')
        #st.line_chart(sma)
    else:
        st.write('Exponential Moving Average')
        st.line_chart(ema)
    
    st.bar_chart(data.Volume)

#Pencarian Data
def dataframe():
    caridata = option_menu(None, ['10 Data','Filter'], icons=['arrow-up-square', 'arrow-down-square'], menu_icon="cast", default_index=0, orientation="horizontal")
    if caridata == '10 Data':
       st.header('10 Data Terkini')
       st.dataframe(data.tail(10))
    else:
       st.header('Filter Data')
       filterdata = pd.read_csv('porto.csv', index_col=[0])
       filterdata = filterdata.rename(columns = {"p": "Level","kode":"Kode","rekom": "Saran","now":"Harga","l":"1YMin","h":"1YMax","hr":"2M","bl":"6M", 
       "opm":"Margin Operasi(%)", "dev":"Deviden PR(%)","adev":"Deviden 5Y","roe": "ROE(%)",
       "pbv": "Nilai Buku","dte": "Rasio UM","eg4": "PhGrow","etr": "Pendapatan","rg":"RevGrow",
       "pm":"Profit Margin", "tcs": "Kas Per Saham", "avol": "AVolume"}).sort_values(['Harga','Level'])
       st.dataframe(filterdata.tail(10))
        
def predict():
    model = st.radio('Pilih Model Komputasi', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.number_input('Prediksi harga saham beberapa hari ke depan?', value=5)
    num = int(num)
    if st.button('Prediksi Harga Saham'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)
        else:
            engine = XGBRegressor()
            model_engine(engine, num)


def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'AKURASI {ceknon(r2_score(y_test, preds))}%')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Hari ke-{day}: {round(i)}')
        day += 1
        
#Screener Grafik Kuadran
def screener():
    screenlevel = option_menu(None, ['>Rp5rb','<Rp5rb','<Rp200','BagiDeviden'], icons=['arrow-up-square', 'arrow-down-square', 'arrow-down-square-fill', 'bullseye'], menu_icon="cast", default_index=0, orientation="horizontal")
   
    st.subheader('Tabular Hasil Screener')

    scr1 = pd.read_csv('porto.csv', index_col=[0])
    scr1 = scr1.fillna(0)
    if screenlevel == '<Rp200':
       st.write('Screener Saham Harga Rentang 50-200')
       scr1=scr1.query("now > 50 and now <= 200 and p<=10 and opm >= 0.1 and roe >= 0.1")
    elif screenlevel == '<Rp5rb':
       st.write('Screener Saham Harga Kurang Dari 5000')
       scr1=scr1.query("now > 200 and now <= 5000 and p<=10 and opm >= 0.1 and roe >= 0.1")
    elif screenlevel == 'BagiDeviden':
        st.write('Screener Rutin Bagi Deviden di atas 5%')
        dev = pd.read_csv('devhunter.csv')
        dev = dev.values.tolist()
        dev = [item for sublist in dev for item in sublist]
        scr1 = scr1.query("kode in @dev")
    else:
       st.write('Screener Saham Harga Lebih Dari 5000')
       scr1=scr1.query("now > 5000 and p<=10 and opm >= 0.1")

    s = scr1.copy()
    scr1 = scr1.rename(columns = {"p": "Level","kode":"Kode","rekom": "Saran","now":"Harga","l":"1YMin","h":"1YMax","hr":"2M","bl":"6M", 
                                  "opm":"Margin Operasi(%)", "dev":"Deviden PR(%)","adev":"Deviden 5Y","roe": "ROE(%)",
                                  "pbv": "Nilai Buku","dte": "Rasio UM","eg4": "PhGrow","etr": "Pendapatan","rg":"RevGrow",
                                  "pm":"Profit Margin", "tcs": "Kas Per Saham", "avol": "AVolume"}).sort_values(['Harga','Level'])
    #st.dataframe(scr1.style.highlight_max(axis=0),hide_index=True)
    st.dataframe(scr1)
    st.subheader('Grafik')
    fig, ax = plt.subplots()
    
    x = s['p']
    y = s['opm']
    kd = s['kode']
    sns.scatterplot(s,x=x, y=y, marker='>')
    plt.xlabel("Level Harga")
    plt.ylabel("Margin Operasi (%)")
    for a,b,c in zip(x,y,kd):
        label = f"{c} {int(b)}%"
        ax.annotate(label,(a,b), xytext=(3, -3),textcoords='offset points',fontsize='7')
    
    st.pyplot(fig)
    
    
if __name__ == '__main__':
    main()
    
st.sidebar.info("Desain Jonaben & modifikasi oleh DwiA")
