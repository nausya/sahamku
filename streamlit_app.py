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
import pytz
from datetime import date
import socket
from csv import writer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Tentukan zona waktu Indonesia
indonesia_timezone = pytz.timezone('Asia/Jakarta')

# Dapatkan waktu saat ini
# datetime.now(indonesia_timezone)

st.sidebar.info('SELAMAT DATANG (Versi Beta)')
st.title('ANALITIK SAHAM INDONESIA') 

def main():
    selected2 = option_menu(None, ["Home", "Cari Data", "Screener", 'Prediksi'], 
                            icons=['house', 'file-earmark-text', 
                            'sliders2-vertical', 'graph-up-arrow'], menu_icon="cast", default_index=0, orientation="horizontal")
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

##########Notasi Saham################
kode = selected_emiten.split(' | ')[0]
n = pd.read_csv('notasi.csv', sep=';')
n = n[(n['Kode'] == kode)]
if n.isna().empty:
   st.text("")
else:  
   n = n['Keterangan Notasi'].str.split('|', expand=False)
   for y in n:
    for x in y:
     st.error(x)
##########End of Notasi Saham################

#ganti value None
def ceknon(x):
    if x is not None:
       x = round(x,2)*100
       x = int(x)
       return x
    elif x == 0:
       return 0
    else:
       return 0 
        #st.error('This is an error', icon="🚨")

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
    roe = info.get('returnOnEquity')
    pery = info.get('forwardPE')
    epsy = info.get('forwardEps')
    pbvy = info.get('priceToBook')
    bvy = info.get('bookValue')
    aksiy = info.get('recommendationKey')
    vol = info.get('averageDailyVolume10Day')
    mcap = info.get('marketCap')
    totshm = info.get('sharesOutstanding')
    bl = info.get('fiftyDayAverage')
    m = info.get('twoHundredDayAverage')
    cash = info.get('totalCash')
    opcash = info.get('operatingCashflow')
    ph = info.get('totalRevenue')
    ut = info.get('totalDebt')
    tcs = info.get('totalCashPerShare')
    screensaham.append({'kode':kode,'skg':skg,'lo':lo,'hi':hi,'om':om,'dev':dev,'roe':roe,
                       'pery':pery,'epsy':epsy,'pbvy':pbvy,'bvy':bvy,'aksiy':aksiy,'vol':vol,'totshm':totshm,
                      'm':m,'bl':bl,'mcap':mcap,'cash':cash,'opcash':opcash,'ph':ph,'ut':ut,'tcs':tcs})
df = pd.DataFrame(screensaham)
df = df.fillna(0)

L52 = df['lo'].values[0]
H52 = df['hi'].values[0]
C = df['skg'].values[0]
D = (H52-L52)/100

if int(D) > 0 :
    P = (C - L52)/D 
else : 
    P = 0
st.subheader(f"Harga terkini Rp {int(C)} berada pada posisi ke-{int(P)} dari ketinggian 100", divider="rainbow")
st.subheader(f"MarginOps : {ceknon(om)}% | DevPR : {ceknon(dev)}% | ROE : {ceknon(roe)}%", divider="rainbow")
 
def mil(x):
    x = x/1000000000
    return x
    
#FINANSIAL
kodef = selected_emiten.split(' | ')[0]
fin = pd.read_csv('Finansial.csv', sep=";")
fin = fin.query("Kode==@kodef")
fin = fin[['EPSRP','BVRP','PER','PBV','Sektor','KodeInd']]
if fin.empty:
   st.error ("KATEGORI PERUSAHAAN BARU MASUK IPO")
else:
   fin = fin.values.tolist()
   fin = [item for sublist in fin for item in sublist]
   skg = df['skg'].values[0]
   lo = df['lo'].values[0]
   eps = df['epsy'].values[0]
   tcs = df['tcs'].values[0]
   bv = df['bvy'].values[0]
   pbv = df['pbvy'].values[0]
   per = df['pery'].values[0]
   ph = df['ph'].values[0]
   ut = df['ut'].values[0]
   sek = fin[4]
   ind = fin[5][:2]
   vol = int(vol) if vol !=None else 0
   dev = int(dev) if dev !=None else 0
   mcap = int(mcap) if mcap !=None else 0
   st.subheader(f"EPS : Rp.{eps} | BV : Rp.{round(bv)} | PBV : {round(pbv,1)} | PER : {round(per)} | Sektor : {sek}", divider="rainbow")
   
   #BENCHMARK
   kodebm = ind
   bm = pd.read_csv('IndexSektor.csv', sep=";")
   bm = bm.query("Kode==@kodebm")
   bm = bm[['Sektor','EPSRp','BVRp','PER','PBV']]
   bm = bm.values.tolist()
   bm = [item for sublist in bm for item in sublist]
   bmeps = bm[1]
   bmbv = bm[2]
   bmpbv = bm[4]
   bmper = bm[3]
   bmsek = bm[0]
   st.subheader(f"STANDAR KINERJA EMITEN SEJENIS \n EPS : Rp.{bmeps} | BV : Rp.{bmbv} | PBV : {bmpbv} | PER : {bmper} | SubSektor : {bmsek}", divider="rainbow")
   #deps = eps - bmeps
   #dbv = bv - bmbv
   #dpbv = pbv/bmpbv
   #dper = "Idealnya PER < 15"

   #RINGKASAN
   st.subheader(f"RINGKASAN PORTOFOLIO", divider="rainbow")
   date = datetime.datetime.now(indonesia_timezone)
   date = date.strftime('%Y-%m-%d %H:%M:%S')
   aksik = 0
   hostname = socket.gethostname()
   ip_address = socket.gethostbyname(hostname)
   id = hostname+"-"+ip_address
   cash = cash/1000000000
   ph = ph/1000000000
   ut = ut/1000000000
   opcash = opcash/1000000000
   vol = vol/1000000
   mcap = mcap/1000000000
   totshm = totshm/1000000000

   dfringkas = {'kode':kode,'date':date,'skg':skg,'1ylo':lo,'2m':int(bl),'6m':int(m),'1yhi':hi,'om(%)':ceknon(om),
                'dev(%)':ceknon(dev),'roe(%)':ceknon(roe),'pos':round(P,0),'cash(M)':round(cash),'opcash(M)':int(opcash),'ph(M)':int(ph),
                'utang(M)':round(ut),'cash/saham(Rp)':round(tcs),'eps(Rp)':eps,'bv(Rp)':round(bv),'pbv':round(pbv,1),
                'per':round(per),'vol(Juta)':round(vol,0),'TotalSaham(M)':round(totshm),'Omzet(M)':round(mcap),'aksiy':aksiy,'aksik':aksik,'user':id}
   
  # with open('aksi.csv', 'a') as f_object:
   #     writer_object = writer(f_object)
    #    writer_object.writerow(dfringkas)
     #   f_object.close()

    

   dfringkas = pd.DataFrame(dfringkas, index = np.arange(1))
   dfringkas = dfringkas.set_index('kode')
   dfringkas['date'] = pd.to_datetime(dfringkas['date'], format='%Y-%m-%d %H:%M:%S')
   st.dataframe(dfringkas)

st.info('Untuk jangka panjang perlu diperhatikan kisaran posisi harga kurang dari 10')


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
    caridata = option_menu(None, ['10 Data','Portofolio','Deviden','Index Per Sektor','Finansial'], icons=['arrow-up-square', 'arrow-down-square'], menu_icon="cast", default_index=0, orientation="horizontal")
    if caridata == '10 Data':
       st.header('10 Data Terkini')
       st.dataframe(data.tail(10))
    elif caridata == 'Deviden':
        st.header("Data Deviden")
        devcum = pd.read_csv('devcumdate.csv', index_col=[0], sep=';')
        st.dataframe(devcum)
    elif caridata == 'Index Per Sektor':
        st.header("Index Per Sektor")
        indsektor = pd.read_csv('IndexSektor.csv', index_col=[0], sep=';')
        st.dataframe(indsektor)
    elif caridata == 'Finansial':
        st.header("Finansial (Milyar Rupiah)")
        keu = pd.read_csv('Finansial.csv', index_col=[0], sep=';')
        st.dataframe(keu)
   #date
    else:
       st.header('Filter Data')
       filterdata = pd.read_csv('porto.csv', index_col=[0], sep=';')
       filterdata[date].unique()
       filterdata = filterdata.rename(columns = {"p": "Posisi","kode":"Kode","aksiy": "Saran","skg":"Harga","lo":"1YMin","hi":"1YMax","bl":"2M","m":"6M", 
       "opm":"Margin Operasi(%)", "dev":"Deviden PR(%)","epsy":"Laba Per Saham","roe": "ROE(%)","pery": "PER(%)",
       "pbvy": "Nilai Buku","bvy": "Harga Dasar","ph": "Pendapatan","ut":"Utang",
       "pm":"Profit Margin", "cash": "Jumlah Kas", "opcash": "Kas Operasional","tcs": "Kas Per Saham", "totshm": "Saham Beredar","mcap": "Omzet","vol": "Volume"}).sort_values(['Harga','Posisi'])
       st.dataframe(filterdata)
        
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
    scr1 = pd.read_csv('porto.csv', sep=';')
   
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
    #scr1 = scr1.reset_index()
    s = scr1.copy()
    scr1 = scr1.rename(columns = {"p": "Posisi","kode":"Kode","rekom": "Saran","now":"Harga","l":"1YMin","h":"1YMax","hr":"2M","bl":"6M", 
                                  "opm":"Margin Operasi(%)", "dev":"Deviden PR(%)","adev":"Deviden 5Y","roe": "ROE(%)",
                                  "pbv": "Nilai Buku","dte": "Rasio UM","eg4": "PhGrow","etr": "Pendapatan","rg":"RevGrow",
                                  "pm":"Profit Margin", "tcs": "Kas Per Saham", "avol": "AVolume"}).sort_values(['Harga','Posisi'])
    #st.dataframe(scr1.style.highlight_max(axis=0),hide_index=True)
    st.dataframe(scr1)
    st.subheader('Grafik')
    fig, ax = plt.subplots()
    
    x = s['p']
    y = s['opm']
    kd = s['kode']
    sns.scatterplot(s,x=x, y=y, marker='>')
    plt.xlabel("Posisi Harga")
    plt.ylabel("Margin Operasi (%)")
    for a,b,c in zip(x,y,kd):
        label = f"{c} {int(b)}%"
        ax.annotate(label,(a,b), xytext=(3, -3),textcoords='offset points',fontsize='7')
    
    st.pyplot(fig)
    
    
if __name__ == '__main__':
    main()
    
#st.sidebar.info("@2024")
