import streamlit as st
import pandas as pd
import yfinance as yf
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



st.title('ANALITIK SAHAM BURSA EFEK INDONESIA')
st.sidebar.info('SELAMAT DATANG')


def main():
    option = st.sidebar.selectbox('Silakan Pilih', ['Grafik','Data', 'Prediksi'])
    if option == 'Grafik':
        tech_indicators()
    elif option == 'Data':
        dataframe()
    else:
        predict()



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
 
## Filter the data
#filtered_data = dataemiten.loc[dataemiten['Kode'] == selected_emiten]
 
## Display the filtered data
st.header(selected_emiten.split(' | ')[1])

option = selected_emiten.split(' | ')[0] + ".JK"

#Display Persentil
detil = yf.Ticker(option)
L52 = detil.info['fiftyTwoWeekLow']
H52 = detil.info['fiftyTwoWeekHigh']
C = detil.info['currentPrice']
D = (H52-L52)/100
P = (C - L52)/D
st.subheader(f"Harga terkini Rp.{{:7,.0f}.format(C)}- berada pada level {round(P)} dari skala 100", divider="rainbow")
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
    option = st.radio('Pilih Teknikal Indikator', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

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
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
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
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Exponential Moving Average')
        st.line_chart(ema)
    
    st.bar_chart(data.Volume)


def dataframe():
    st.header('10 Data Terkini')
    st.dataframe(data.tail(10))



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
    st.text(f'AKURASI \nr2_score: {round(r2_score(y_test, preds)*100)}% \
            \nMAE: {round(mean_absolute_error(y_test, preds))}%')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Hari ke-{day}: {round(i)}')
        day += 1


if __name__ == '__main__':
    main()
    
st.sidebar.info("Desain Jonaben & modifikasi oleh DwiA")
