import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import yfinance as yf
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
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
import os



####### TOKEN GITHUB
# Dapatkan token GitHub dari variabel lingkungan
github_token = os.environ.get('GITHUB_TOKEN')

######Halaman Utama
st.set_page_config(page_title="Sahamku", layout="wide")
st.header('ANALITIK SAHAM INDONESIA')
######End of Halaman Utama
import streamlit as st


###### FUNGSI MENU #############
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
###### END OF FUNGSI MENU #############
@st.cache_resource
############# KUMPULAN FUNGSI ######
#Percentil
def pentil(min,max,c):
    p = np.interp(c, [min, max], [0, 100])
    return round(p)
    
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
############# fungsi delta ######
def d(a, b):
    return (a - b)
def dpros(a, b):
    return (a / b) * 100
def dx(a, b):
    return (a / b)
def mil(x):
    x = x/1000000000
    return x
def digit(angka):
    if angka >= 10**12:
        return f"{angka / 10**12:.1f} T"
    elif angka >= 10**9:
        return f"{angka / 10**9:.1f} M"
    elif angka >= 10**6:
        return f"{angka / 10**6:.1f} Jt"
    elif angka >= 10**3:
        return f"{angka / 10**3:.1f} Rb"
    elif angka <= -10**12:
        return f"{angka / 10**12:.1f} T"
    elif angka <= -10**9:
        return f"{angka / 10**9:.1f} M"
    elif angka <= -10**6:
        return f"{angka / 10**6:.1f} Jt"
    elif angka <= -10**3:
        return f"{angka / 10**3:.1f} Rb"
    else:
        return str(angka)
######### END of FUngsi DELTA ########################

####### FUNGSI AMBIL DATA SAHAM ############
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df
####### END OF FUNGSI AMBIL DATA SAHAM ############

######################## Tentukan zona waktu Indonesia
indonesia_timezone = pytz.timezone('Asia/Jakarta')
date = datetime.datetime.now(indonesia_timezone)
date = date.strftime('%Y-%m-%d')# %H:%M:%S
# Dapatkan waktu saat ini
# datetime.now(indonesia_timezone)

st.sidebar.info('SELAMAT DATANG (Versi Beta)')
#######AMBIL KODE EMITEN DARI CSV

######## PILIH DATA EMITEN PADA MENU SEBELAH KIRI
dataemiten = pd.read_csv('kodesaham.csv').sort_values('Kode')
emiten = dataemiten['Kode'] + ' | ' + dataemiten['Nama Perusahaan']
selected_emiten = st.sidebar.selectbox('Pilih Emiten:', emiten)
namatampil = selected_emiten.split(' | ')[1]
option = selected_emiten.split(' | ')[0] + ".JK"
kodesaja = selected_emiten.split(' | ')[0]
########Proses sidebar data
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
######## End Proses sidebar data

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
    vole = info.get('volume')
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
    #der = info.get('debtToEquity')
    screensaham.append({'kode':kode,'skg':skg,'lo':lo,'hi':hi,'om':om,'dev':dev,'roe':roe,
                       'pery':pery,'epsy':epsy,'pbvy':pbvy,'bvy':bvy,'aksiy':aksiy,'vol':vol,'totshm':totshm,
                      'm':m,'bl':bl,'mcap':mcap,'cash':cash,'opcash':opcash,'ph':ph,'ut':ut,'tcs':tcs})
#kodek = df['kode'].values[0]
df = pd.DataFrame(screensaham)
df = df.fillna(0)
BL = df['bl'].values[0]
M = df['m'].values[0]
L52 = df['lo'].values[0]
H52 = df['hi'].values[0]
C = df['skg'].values[0]
##### BILA DATA LIVE BELUM TERSEDIA skg;lo;bl;m;hi
HgPor = pd.read_csv('porto.csv', index_col=[0], sep=';')
HgPor = HgPor.query("kode ==@kodesaja")
HPS = HgPor['skg'].values[0]
HPL = HgPor['lo'].values[0]
HPB = HgPor['bl'].values[0]
HPM = HgPor['m'].values[0]
HPH = HgPor['hi'].values[0]
Sahamedar = HgPor['totshm'].values[0]
##################### END OF BILA DATA LIVE BELUM TERSEDIA
if C == 0 or L52==0 or H52==0 or BL==0 or M==0:
    C = data.tail(1)
    C = C['Adj Close'].values[0]
    L52 = HPL
    H52 = HPH
    BL = HPB
    M = HPM
P1 = pentil(L52,BL,C)
P2 = pentil(L52,M,C)
P3 = pentil(L52,H52,C)
if C <=50:
  P1 = 50
  P2 = 50
  P3 = 50
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
user = hostname+"-"+ip_address
#st.subheader(f"Harga terkini Rp{int(C)} dimana setahun terakhir tingkat harga berada pada posisi ke-{P3} dari ketinggian 100", divider="rainbow")
################# TAMPILKAN NAMA EMITEN
col1, col2 = st.columns([1, 1])
with col1:
    st.title(f'_:blue[{namatampil}]_')
    ###### KONTAINER TABS SIMULASI #############
    tab1, tab2, tab3, tab4 = st.tabs(['Halaman Simulasi','Simulasi Beli','Simulasi Jual','MinMax'])                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    with tab1:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
       st.write('Selamat Datang di Halaman Simulasi')
    with tab2:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
       ########PROSES SIMPAN SIMULASI                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
       volbeli = st.number_input('Banyaknya saham :', value=1000, step=100)  
       volbeli = round(volbeli/100)
       st.write('Jumlah pembelian saham : ', volbeli, 'lot') 
       volbeli = volbeli * 100
       volbeli = int(volbeli)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
       if st.button('simpan') and volbeli >= 100:
            dfaksi = pd.read_csv('aksi.csv', sep=";")
            date = date
            brs = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            aksik = 'buy'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
            total = C * volbeli                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            brs.append({'date': date, 'kode': kodesaja, 'skg': C, 'p1': P1, 'p2': P2, 'p3': P3, 'vol': volbeli, 'aksiy': aksiy,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                        'aksik': aksik, 'total': total, 'user': user})                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            n = pd.DataFrame(brs)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
            dfaksi = pd.concat([dfaksi, n], ignore_index=True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
            dfaksi.to_csv('aksi.csv', sep=";", index=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 


            st.write(f'Simulasi pembelian saham {namatampil} pada tanggal {date} sebanyak {digit(volbeli)} lembar berhasil disimpan. Total transaksi adalah Rp. {digit(round(total))},-')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
       else:
           st.error('Jumlah Minimal Beli 1 Lot Saham')
    with tab3:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        dfjual = pd.read_csv('aksi.csv', sep=";", usecols=['kode','skg','vol','aksik','total'])
        dfjual = dfjual.query("kode == @kodesaja and aksik=='buy'")
        #dfjual
        if dfjual.empty:
           st.write("Belum Terdapat Simulasi Penjualan Saham Ini")
        else:
            hb = dfjual['skg'].mean()
            vj = dfjual['vol'].sum()
            tb = dfjual['total'].sum()
            tj = C * vj
            prosj = dpros(tj,tb) - 100
            mj = tj - tb
            lj = round(vj/100)
            st.metric(f"Margin Simulasi Penjualan Saham {kodesaja} : \n\nHarga Beli @Rp{round(hb)} sejumlah", f"{lj} Lot", f"{digit(mj)} ({round(prosj,1)}%)") 

    with tab4:
        mm = yf.Ticker(option).history(period="max")                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        mm = mm.describe()
        mm = mm.astype(int)
        st.dataframe(mm)
        
######End of Halaman Utama
with col2:

    if C <= 50 or L52 <=50:
       hg = st.slider('Harga Terkini', 0, round(H52), round(C),  disabled=False, step=5)
       marg = (hg - C)/hg * 100
       marg = round(marg,1)
       st.write('Margin<span style="font-size: 30px;">', marg, '</span>%', unsafe_allow_html=True)
        
    else:
       hg = st.slider('Harga Terkini', round(L52), round(H52), round(C),  disabled=False, step=5)
       marg = (hg - C)/hg * 100
       marg = round(marg,1)
       #st.write("Margin", marg, '%')
       st.write('Margin<span style="font-size: 30px;">', marg, '</span>%', unsafe_allow_html=True)
################# END OF TAMPILKAN NAMA EMITEN
##########Notasi Saham################
kode = selected_emiten.split(' | ')[0]
n = pd.read_csv('notasi.csv', sep=';')
nk = pd.read_csv('notasi-khusus.csv', sep=';')
n = pd.concat([n, nk], axis=0)
nota = n.copy()
nota = nota.drop_duplicates()
n = n[(n['Kode'] == kode)]
if n.isna().empty:
   st.text("")
else:  
   n = n['Keterangan Notasi'].str.split('|', expand=False)
   for y in n:
    for x in y:
     st.error(x)
##########End of Notasi Saham################
###### CHART GAUGE
plot_bgcolor = "lightcyan"
quadrant_colors = [plot_bgcolor, "red", "#f2a529", "#eff229", "#85e043", "#2bad4e"] 
quadrant_text = ["", "<b>Kuat Jual</b>", "<b>Jual</b>", "<b>Netral</b>", "<b>Beli</b>", "<b>Kuat Beli</b>"]
n_quadrants = len(quadrant_colors) - 1
kodechart = kodesaja

# Data untuk 3 variabel
variables = [
    {"current_value": P1, "jangka": '< 2 Bulan', "jenis": '- Gorengan -'},
    {"current_value": P2, "jangka": '2-12 Bulan', "jenis": '- Cemilan -'},
    {"current_value": P3, "jangka": '> 1 Tahun', "jenis": '- Tanam Jati -'}
]

# Looping untuk membuat 3 chart gauge
figs = []
for variable in variables:
    current_value = variable["current_value"]
    jangka = variable["jangka"]
    jenis = variable["jenis"]
    min_value = 0
    max_value = 100
    hand_length = np.sqrt(2) / 4.5
    hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

    fig = go.Figure(
         data=[
             go.Pie(
                  values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
                  rotation=90,
                  hole=0.5,
                  marker_colors=quadrant_colors,
                  text=quadrant_text,
                  textinfo="text",
                  hoverinfo="skip",
              ),
          ],
          layout=go.Layout(
              showlegend=False,
              margin=dict(b=0,t=10,l=10,r=10),
              width=300,
              height=300,
              paper_bgcolor=plot_bgcolor,
              annotations=[
                  go.layout.Annotation(
                      text=f"<b>{current_value}</b>",
                      x=0.5, xanchor="center", xref="paper",
                      y=0.6, yanchor="bottom", yref="paper",
                      showarrow=False,
                      font=dict(size=20) # Ubah ukuran font di sini
                  ),
                  go.layout.Annotation(
                      text=f"<b>{kodechart}</b>",
                      x=0.5, xanchor="center", xref="paper",
                      y=0.08, yanchor="bottom", yref="paper",
                      showarrow=False,
                      font=dict(size=35) # Ubah ukuran font di sini
                  ),
              go.layout.Annotation(
                text=f"<b>Indikator Harga {jangka} <br>{jenis}</b>",
                x=0.5, xanchor="center", xref="paper",
                y=0.3, yanchor="bottom", yref="paper",
                showarrow=False)
           ],
           shapes=[
               go.layout.Shape(
                   type="circle",
                   x0=0.48, x1=0.52,
                   y0=0.48, y1=0.52,
                   fillcolor="grey",
                   line_color="grey",
               ),
               go.layout.Shape(
                   type="line",
                   x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                   y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                   line=dict(color="grey", width=2)
               )
            ]
        )
    )
    figs.append(fig)
# Membagi layout menjadi tiga kolom
col1, col2, col3 = st.columns([1, 1, 1])

# Menampilkan chart ke dalam masing-masing kolom
with col1:
    figs[0]

with col2:
    figs[1]

with col3:
    figs[2]
##################################### END OF CHART GAUGE
P = P3
#####################  FINANSIAL
kodef = selected_emiten.split(' | ')[0]
fin = pd.read_csv('Finansial.csv', sep=";")
fin = fin.query("Kode==@kodef")
#fin
fin = fin[['EPSRP','BVRP','PER','PBV','Sektor','KodeInd','Utang','Proft']]
if fin.empty:
   st.error ("KATEGORI PERUSAHAAN BARU MASUK IPO")
else:
   fin = fin.values.tolist()
   fin = [item for sublist in fin for item in sublist]
   skg = df['skg'].values[0]
   lo = df['lo'].values[0]
   eps = fin[0]
   tcs = df['tcs'].values[0]
   bv = df['bvy'].values[0]
   pbv = df['pbvy'].values[0]
   per = df['pery'].values[0]
   ph = df['ph'].values[0]
   ut = df['ut'].values[0]
   utlap = fin[6] * 1000000000
   sek = fin[4]
   ind = fin[5][:2]
   prf = fin[7] * 1000000000
   per = per if per !='Infinity' else 0
   vol = int(vol) if vol !=None else 0
   vole = int(vole) if vole !=None else 0
   dev = dev if dev !=None else 0
   mcap = int(mcap) if mcap !=None else 0

   #BENCHMARK
   kodebm = ind
   bm = pd.read_csv('IndexSektor.csv', sep=";")
   bm = bm.query("Kode==@kodebm")
   bm = bm[['Sektor','EPSRp','BVRp','PER','PBV','DER']]
   bm = bm.values.tolist()
   bm = [item for sublist in bm for item in sublist]
   bmeps = round(bm[1])
   bmbv = bm[2]
   bmpbv = bm[4]
   bmper = bm[3]
   bmder = bm[5]
   bmsek = bm[0]
   deps = round(d(eps, bmeps)) 
   dbv = round(d(bv, bmbv))
   dpbv = round(d(bmpbv, pbv))
   dper = round(d(bmper, per))

   dprf = eps * Sahamedar
   dvol = round(d(vol, vole))
   ddev = 0
   dom = 0
   utun = digit(cash + opcash - ut)
   dtcs = str(round(dx(tcs, C),1)) + "X"
   dut = digit(mcap)
   tunai = digit(cash)
   volshm = digit(vol)
   # metric1 
   col1, col2, col3, col4, col5 = st.columns(5)
   col1.metric("Laba Per Saham(Rp)", round(eps), deps)
   col2.metric("Harga Buku(Rp)", round(bv), dbv)
   col3.metric("Nilai Buku", round(pbv,1), dpbv)
   col4.metric("Rata2 Vol Saham", volshm, digit(dvol))
   col5.metric("Tunai Per Saham(Rp)", round(tcs), dtcs)
   col6, col7, col8, col9, col10 = st.columns(5)
   col6.metric("Deviden", str(ceknon(dev)) + "%", 0)
   col7.metric("Kas Operasional(Rp)", digit(opcash), dom)
   col8.metric("Uang Tunai(Rp)", tunai, utun)
   col9.metric("Laba(Rp)", digit(round(prf)), digit(dprf))
   col10.metric("Utang(Rp)", digit(ut), digit(utlap))
   st.subheader("", divider="rainbow")

   st.subheader(f"STANDAR KINERJA \n EPS : Rp.{bmeps} | BV : Rp.{round(bmbv)} | PBV : {bmpbv} | PER : {round(bmper)} | DER : {bmder} | SEKTOR : {bmsek}", divider="rainbow")

   ################## BENCHMARK DETIL
   st.subheader(f"KINERJA EMITEN SEJENIS")
   col1, col2, col3 = st.columns([2, 1, 4])
   with col1:
       bmd = pd.read_csv('Finansial.csv', sep=';', usecols=['Kode','KodeInd','EPSRP','BVRP','PER','PBV','DER','ROA(%)','ROE(%)','NPM(%)']).sort_values('Kode')
       bmd = bmd.query("KodeInd == @fin[5]")
       P = pd.read_csv('porto.csv', sep=';',usecols=['kode','p'])
       bmd = pd.merge(bmd, P, left_on='Kode', right_on='kode', how='inner')
       bmd = bmd[['Kode','P','EPSRP','BVRP','PER','PBV','DER','ROA(%)','ROE(%)','NPM(%)']]
       g = bmd.copy()
       bmd = bmd.set_index('Kode')
       bmd
       
   with col2:
       pawal, pakhir = st.slider('Posisi', min_value=0, max_value=100, value=(0, 100))  
       pilkin = st.radio('Posisi Harga Terhadap : ', ['Laba Per Saham', 'Harga Buku', 'PER', 'Nilai Buku', 'Rasio Utang', 'ROA', 'ROE', 'NPM'])
   with col3:
        fig, ax = plt.subplots()
        g = g.query("p>=@pawal and p<=@pakhir")
        x = g['p']
        if pilkin == 'Laba Per Saham':
            y = g['EPSRP']
            ylabel = 'Laba Per Saham'
        elif pilkin == 'Harga Buku':
            y = g['BVRP']
            ylabel = 'Harga Buku'
        elif pilkin == 'PER':
            y = g['PER']
            ylabel = 'PER'
        elif pilkin == 'Nilai Buku':
            y = g['PBV']
            ylabel = 'Nilai Buku'
        elif pilkin == 'Rasio Utang':
            y = g['DER']
            ylabel = 'Rasio Utang Modal'
        elif pilkin == 'ROA':
            y = g['ROA(%)']
            ylabel = 'Rasio Pengembalian Aset (%)'
        elif pilkin == 'ROE':
            y = g['ROE(%)']
            ylabel = 'Rasio Pengembalian Modal (%)'
        else:
            y = g['NPM(%)']
            ylabel = 'Margin Operasi (%)'
        kd = g['Kode']
        sns.scatterplot(g,x=x, y=y, marker='>')
        plt.xlabel("Posisi Harga")
        plt.ylabel(f"{ylabel}")
        for a,b,c in zip(x,y,kd):
            if pilkin in ('Tunai Per Saham', 'Nilai Buku'):
                label = f"{c} {round(b,1)}X"
            else:
                label = f"{c} {round(b,1)}"
            ax.annotate(label,(a,b), xytext=(3, -3),textcoords='offset points',fontsize='7')
        
        st.pyplot(fig)
   st.subheader("", divider="rainbow")
   ################## END OF BENCHMARK DETIL

   ##################RINGKASAN
   st.subheader(f"RINGKASAN PORTOFOLIO")

   aksik = 0
   cash = cash/1000000000
   ph = ph/1000000000
   ut = ut/1000000000
   opcash = opcash/1000000000
   vol = vol/1000000
   mcap = mcap/1000000000
   totshm = totshm/1000000000

   dfringkas = {'Kode':kode,'Tanggal':date,'Harga':C,'1YMin':L52,'2Mon':int(BL),'6Mon':int(M),'1YMax':H52,'Margin Operasi(%)':ceknon(om),
                'Deviden(%)':ceknon(dev),'ROE(%)':ceknon(roe),'Posisi':round(P,0),'Uang Tunai(M)':round(cash),'Kas Operasional(M)':int(opcash),'Pendapatan(M)':int(ph),
                'Utang(M)':round(ut),'Kas Per Saham(Rp)':round(tcs),'EPS(Rp)':eps,'Harga Buku(Rp)':round(bv),'Nilai Buku':round(pbv,1),
                'PER':round(per),'Volume(Juta)':round(vol,0),'TotalSaham(M)':round(totshm),'Omzet(M)':round(mcap),'Saran':aksiy}#,'Aksi':aksik,'User':id}    

   dfringkas = pd.DataFrame(dfringkas, index = np.arange(1))
   dfringkas = dfringkas.set_index('Kode')
   st.dataframe(dfringkas)
   st.subheader("", divider="rainbow")
   ##################END OF RINGKASAN



st.info('Untuk jangka panjang perlu diperhatikan kisaran posisi harga kurang dari 10')

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

#################### CARI DATA ###############
def dataframe():
    caridata = option_menu(None, ['Simulasi','10 Data','Fundamental','Index Per Sektor'], icons=['arrow-up-square', 'arrow-down-square'], menu_icon="cast", default_index=0, orientation="horizontal")
    if caridata == 'Simulasi':
        simul = pd.read_csv('aksi.csv', index_col=[0], sep=';')
        #simul = simul.rename(columns = {"date": "Tanggal", "p1": "Pendek", "p2": "Menengah", "p3": "Panjang", "vol": "Volume", "aksiy": "Saran", "aksik": "Aksi", "total": "Jumlah Rupiah", "user": "Pengguna", "kode": "Emiten", "skg": "Harga Beli"})
        tab1, tab2 = st.tabs(["Beli", "Jual"])
        with tab1:
            st.write("Simulasi Beli")
            simulb = simul.query("aksik=='buy'")
            st.dataframe(simulb)
        with tab2:
            st.write("Simulasi Jual")
            simulj = simul.query("aksik=='sell'")
            st.dataframe(simulj)
    elif caridata == '10 Data':
       st.header('10 Data Terkini')
       st.dataframe(data.tail(10))
    elif caridata == 'Fundamental':
        ###### KONTAINER TABS #############
        tab1, tab2, tab3 = st.tabs(['Portofolio','Deviden','Finansial'])
        st.write('Last Update : 4 Mei 2024')
        with tab1:
           st.write('Filter Data')
           
           filterdata = pd.read_csv('porto.csv', index_col=[0], sep=';')
           #tgl = filterdata['date'].values[0]
           #tgl = tgl[8:10] + "/" + tgl[5:7]+ "/" + tgl[0:4]
           #"Last Update : " + tgl
           filterdata = filterdata.rename(columns = {"date": "Tanggal", "p": "Posisi","kode":"Kode","aksiy": "Saran","skg":"Harga","lo":"1YMin","hi":"1YMax","bl":"2M","m":"6M", 
           "om":"Margin Operasi", "dev":"Deviden PR","epsy":"Laba Per Saham","roe": "ROE","pery": "PER",
           "pbvy": "Nilai Buku","bvy": "Harga Dasar","ph": "Pendapatan","ut":"Utang",
           "pm":"Profit Margin", "cash": "Jumlah Kas", "opcash": "Kas Operasional","tcs": "Kas Per Saham", "totshm": "Saham Beredar","mcap": "Omzet","vol": "Volume"})
           st.dataframe(filterdata)
        with tab2:
            st.write("Data Deviden")
            devcum = pd.read_csv('devcumdate.csv', index_col=[0], sep=';')
            st.dataframe(devcum)
        with tab3:
            st.write("Finansial (Milyar Rupiah)")
            keu = pd.read_csv('Finansial.csv', index_col=[0], sep=';')
            st.dataframe(keu)

    else:
        st.header("Index Per Sektor")
        indsektor = pd.read_csv('IndexSektor.csv', index_col=[0], sep=';')
        st.dataframe(indsektor)
  
        ###### KONTAINER TABS #############
      
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
        
###############Screener Grafik Kuadran
def screener():
    screenlevel = option_menu(None, ['Fraksi Harga','LQ45','KOMPAS100','BagiDeviden','Industri'], icons=['arrow-up-square', 'arrow-down-square', 'arrow-down-square-fill', 'bullseye'], menu_icon="cast", default_index=0, orientation="horizontal")
   

    scr1 = pd.read_csv('porto.csv', sep=';', index_col=False)
    
    scr1['tun'] = scr1['tcs']/scr1['skg']
    scr1['tun'] = round(scr1['tun'],1)
    scr1 = scr1.fillna(0)
    if screenlevel == 'Fraksi Harga':
        pilihhg = st.selectbox('Pilih Fraksi Harga :', ['Semua Harga','Kurang Dari Rp200','Kurang Dari Rp5000','Lebih Dari Rp5000','Notasi Khusus'])
        if pilihhg == 'Semua Harga':
            st.subheader('Screener Saham Semua Harga')

        elif pilihhg == 'Notasi Khusus':
            st.subheader('Screener Saham Emiten Dengan Notasi Khusus')
            notkus = nota['Kode'].values.tolist()
            scr1 = scr1.query("kode in @notkus")
            
        elif pilihhg == 'Kurang Dari Rp5000':
            st.subheader('Screener Saham Dengan Fraksi Harga Kurang Dari 5000')
            scr1=scr1.query("skg > 200 and skg <= 5000")
           #scr1=scr1.query("skg > 200 and skg <= 5000 and p>=@awal and p<=@akhir and om>=@omawal and om<=@omakhir and dev>=@devawal and dev<=@devakhir and roe>=@roeawal and roe<=@roeakhir and tun>=@tunawal and tun<=@tunakhir and pbvy>=@nbawal and pbvy<=@nbakhir")

        elif pilihhg == 'Lebih Dari Rp5000':
            st.subheader('Screener Saham Dengan Fraksi Harga Lebih Dari 5000')
            scr1 = scr1.query("skg > 5000")
            #scr1 = scr1.query("skg > 5000 and p>=@awal and p<=@akhir and om>=@omawal and om<=@omakhir and dev>=@devawal and dev<=@devakhir and roe>=@roeawal and roe<=@roeakhir and tun>=@tunawal and tun<=@tunakhir and pbvy>=@nbawal and pbvy<=@nbakhir")

        else:
            st.subheader('Screener Saham Dengan Fraksi Harga Rentang 50 - 200')
            scr1=scr1.query("skg > 50 and skg<= 200")
           #scr1=scr1.query("skg > 50 and skg<= 200 and p>=@awal and p<=@akhir and om>=@omawal and om<=@omakhir and dev>=@devawal and dev<=@devakhir and roe>=@roeawal and roe<=@roeakhir and tun>=@tunawal and tun<=@tunakhir and pbvy>=@nbawal and pbvy<=@nbakhir")

                
    elif screenlevel == 'BagiDeviden':
        st.subheader('Screener Rutin Bagi Deviden di atas 5%')
        dev = pd.read_csv('devhunter.csv')
        dev = dev.values.tolist()
        dev = [item for sublist in dev for item in sublist]
        scr1 = scr1.query("kode in @dev")
        #scr1 = scr1.query("kode in @dev and p>=@awal and p<=@akhir and om>=@omawal and om<=@omakhir and dev>=@devawal and dev<=@devakhir and roe>=@roeawal and roe<=@roeakhir and tun>=@tunawal and tun<=@tunakhir and pbvy>=@nbawal and pbvy<=@nbakhir")

    elif screenlevel == 'LQ45':
        st.subheader('Screener Saham LQ45')
        LQ = pd.read_csv('LQ45.csv')
        LQ = LQ.values.tolist()
        LQ = [item for sublist in LQ for item in sublist]
        scr1 = scr1.query("kode in @LQ")
        #scr1 = scr1.query("kode in @LQ and p>=@awal and p<=@akhir and om>=@omawal and om<=@omakhir and dev>=@devawal and dev<=@devakhir and roe>=@roeawal and roe<=@roeakhir and tun>=@tunawal and tun<=@tunakhir and pbvy>=@nbawal and pbvy<=@nbakhir")

    elif screenlevel == 'KOMPAS100':
        st.subheader('Screener Saham KOMPAS100')
        KOMPAS100 = pd.read_csv('KOMPAS100.csv')
        KOMPAS100 = KOMPAS100.values.tolist()
        KOMPAS100 = [item for sublist in KOMPAS100 for item in sublist]
        scr1 = scr1.query("kode in @KOMPAS100")
        #scr1 = scr1.query("kode in @KOMPAS100 and p>=@awal and p<=@akhir and om>=@omawal and om<=@omakhir and dev>=@devawal and dev<=@devakhir and roe>=@roeawal and roe<=@roeakhir and tun>=@tunawal and tun<=@tunakhir and pbvy>=@nbawal and pbvy<=@nbakhir")

    else:
        ind = pd.read_csv('Finansial.csv', sep=';', usecols=['Sektor', 'SubIndustri', 'Kode']).sort_values('SubIndustri')

        sektor = ind['Sektor']
        sektor = sektor.drop_duplicates()
        pilihsek = st.selectbox('Pilih Sektor :', sektor)
        subsek = ind.query("Sektor == @pilihsek")
        subsek = subsek['Kode']
        subsek = subsek.values.tolist()
        scr1 = scr1.query("kode in @subsek")

  
    scr2 = scr1.copy()
    scr2 = scr2.style.applymap(lambda x: 'background-color: pink' if x in nota['Kode'].values else '')
    
    scr2
    s = scr1.copy()
    ########### minmax slider ################
    dfmm = scr1.copy()
    dfmm = dfmm[['p','om','dev','roe','pbvy','tun']]
    dfmax = dfmm.max()
    maxp = round(dfmax['p'])
    maxom = round(dfmax['om'])
    maxdev = round(dfmax['dev'])
    maxroe = round(dfmax['roe'])
    maxnb = round(dfmax['pbvy'])
    maxtun = round(dfmax['tun'])
    dfmin = dfmm.min()
    minp = round(dfmin['p'])
    minom = round(dfmin['om'])
    mindev = round(dfmin['dev'])
    minroe = round(dfmin['roe'])
    minnb = round(dfmin['pbvy'])
    mintun = round(dfmin['tun'])
    ########### end of minmax slider ################
    col1, col2 = st.columns([1,1], gap="large")
    with col1:
        st.markdown('<span style="background-color: #ffc0cb; padding: 3px;">KODE</span> : Emiten Dengan Notasi Khusus', unsafe_allow_html=True)
    with col2:
        st.caption('Last Update : 4 Mei 2024')
    st.write('Filter Berdasarkan :')
    col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1], gap="large")
    with col1:
        awal, akhir = st.slider('Posisi', min_value=0, max_value=100, value=(0, 100))   
    with col2:
        omawal, omakhir = st.slider('Margin Operasi', min_value=0, max_value=100, value=(0, 100)) 
        omawal = omawal
        omakhir = omakhir
       
    with col3:
        devawal, devakhir = st.slider('Deviden', min_value=0, max_value=100, value=(0, 100)) 
        devawal = devawal
        devakhir = devakhir
      
    with col4:
        roeawal, roeakhir = st.slider('Return on Equity', min_value=0, max_value=100, value=(0, 100)) 
        roeawal = roeawal
        roeakhir = roeakhir
       
    with col5:
        tunawal, tunakhir = st.slider('Tunai Per Saham (X)', min_value=mintun, max_value=maxtun, value=(mintun, maxtun))
    
    with col6:
        nbawal, nbakhir = st.slider('Nilai Buku (X)', min_value=0, max_value=1000, value=(0, 1000))        
       
    
    st.subheader('Grafik')
    pilgra = st.radio('Posisi Harga Terkini Terhadap : ', ['Margin Operasi', 'Rasio Bayar Deviden', 'Tunai Per Saham', 'Return on Equity', 'Nilai Buku'])
    fig, ax = plt.subplots()
    s = s.query("p>=@awal and p<=@akhir and om>=@omawal and om<=@omakhir and dev>=@devawal and dev<=@devakhir and roe>=@roeawal and roe<=@roeakhir and tun>=@tunawal and tun<=@tunakhir and pbvy>=@nbawal and pbvy<=@nbakhir")   
    x = s['p']
    if pilgra == 'Margin Operasi': 
        y = s['om']
        ylabel = 'Margin Operasi'
    elif pilgra == 'Rasio Bayar Deviden':
        y = s['dev']
        ylabel = 'Rasio Bayar Deviden'
    elif pilgra == 'Tunai Per Saham':
        y = s['tun']
        ylabel = 'Tunai Per Saham (X)'
    elif pilgra == 'Nilai Buku':
        y = s['pbvy']
        ylabel = 'Nilai Buku (X)'
    else:
        y = s['roe']
        ylabel = 'Return on Equity'
    kd = s['kode']
    sns.scatterplot(s,x=x, y=y, marker='>')
    plt.xlabel("Posisi Harga")
    plt.ylabel(f"{ylabel}")
    for a,b,c in zip(x,y,kd):
        if pilgra in ('Tunai Per Saham', 'Nilai Buku'):
            label = f"{c} {round(b,1)}X"
        else:
            label = f"{c} {round(b,1)}"
        ax.annotate(label,(a,b), xytext=(3, -3),textcoords='offset points',fontsize='7')
    
    st.pyplot(fig)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
if __name__ == '__main__':
    main()
    
#st.sidebar.info("@2024")
