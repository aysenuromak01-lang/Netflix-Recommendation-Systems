import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Netflix Öneri Sistemi", page_icon="🎬", layout="centered")

st.title("🎬 Netflix Yapay Zeka Tabanlı Öneri Motoru")
st.write("Kaggle orijinal Netflix Prize verileri ve Pearson Korelasyonu algoritması kullanılarak geliştirilmiştir.")

# Kayıtlı varlıkları önbelleğe alarak güvenli yükleme
@st.cache_resource
def load_netflix_data():
    try:
        matrix = joblib.load("netflix_real_corr.pkl")
        movies = pd.read_csv("active_movies.csv")
        return matrix, movies
    except Exception as e:
        return None, None

corr_matrix, movies_df = load_netflix_data()

if corr_matrix is not None and movies_df is not None:
    # 1. Kullanıcı Arayüzü Seçim Alanı
    movie_list = sorted(movies_df['Movie_Title'].astype(str).values)
    selected_movie = st.selectbox("İzlediğiniz ve beğendiğiniz bir Netflix filmi seçin:", options=movie_list)
    
    # 2. Öneri Algoritmasının Tetiklenmesi
    if st.button("Benzer Filmleri Getir"):
        with st.spinner("İzleyici veri tabanı analiz ediliyor, benzer zevklere sahip kitleler eşleştiriliyor..."):
            try:
                # Seçilen filmin diğer tüm filmlerle olan korelasyon serisini al
                target_correlations = corr_matrix[selected_movie].dropna()
                
                # Skorları en yüksekten en düşüğe doğru sırala
                recommendations = target_correlations.sort_values(ascending=False)
                
                # Seçilen filmin kendisini öneri listesinden çıkar
                recommendations = recommendations.drop(selected_movie, errors='ignore')
                
                st.subheader(f"✨ '{selected_movie}' Yapımını Beğenenlerin Tercih Ettiği Diğer Filmler:")
                
                # En yüksek korelasyona sahip ilk 5 filmi ekrana yazdır
                if len(recommendations) == 0:
                    st.warning("Bu içerikle eşleşen yeterli ortak izleyici verisi bulunamadı.")
                else:
                    for rank, (movie_title, score) in enumerate(recommendations.head(5).items(), start=1):
                        # Korelasyon skorunu yüzdeye çevirerek daha anlaşılır yapıyoruz
                        st.success(f"🎥 **{rank}. Öneri:** {movie_title} — *(İzleyici Eğilim Uyumu: %{score*100:.1f})*")
                        
            except Exception as e:
                st.error(f"Öneri üretilirken teknik bir hata oluştu: {e}")
else:
    st.error("Gerekli model dosyaları (`netflix_real_corr.pkl` veya `active_movies.csv`) klasörde bulunamadı! Lütfen önce Jupyter Notebook dosyanızı çalıştırarak modelleri eğitin.")