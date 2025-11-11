# ============================================================
# 🥗 STREAMLIT APP: Prediksi Kecukupan Nutrisi Harian
# ============================================================

import streamlit as st
import pandas as pd
import joblib

# ============================================================
# 🔹 LOAD MODEL DAN DATASET
# ============================================================

MODEL_PATH = "random_forest_healthscore_CLASSIFIER.joblib"
DATA_PATH = "nutrition.csv"

# Load model
model = joblib.load(MODEL_PATH)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Normalisasi nama kolom (jaga-jaga)
df.columns = df.columns.str.strip().str.lower()

# ============================================================
# 🔹 KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Prediksi Kecukupan Nutrisi Harian",
    layout="wide",
    page_icon="🥗",
)

st.title("🥗 Prediksi Kecukupan Nutrisi Harian")
st.markdown("Masukkan data dirimu dan makanan yang dikonsumsi hari ini untuk memprediksi **skor kesehatan (1–5)** berdasarkan model AI Random Forest.")

# ============================================================
# 🔹 INPUT DATA PENGGUNA
# ============================================================

st.sidebar.header("🧍‍♂️ Data Pengguna")

age = st.sidebar.number_input("Umur", 10, 100, 25)
gender = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"])
weight = st.sidebar.number_input("Berat Badan (kg)", 30, 200, 70)
height = st.sidebar.number_input("Tinggi Badan (cm)", 100, 220, 170)

st.sidebar.markdown("---")
st.sidebar.caption("👆 Isi data sebelum memilih makanan di bawah.")

# ============================================================
# 🔹 FITUR PENCARIAN MAKANAN
# ============================================================

st.subheader("🍛 Pilih Makanan/Minuman")

search_query = st.text_input("Cari nama makanan atau minuman:")
filtered_df = df[df["name"].str.contains(search_query, case=False, na=False)] if search_query else df

# Tampilkan hasil pencarian
st.dataframe(filtered_df[["name", "calories", "fat", "proteins", "carbohydrate"]].head(10))

# ============================================================
# 🔹 PENAMBAHAN MAKANAN TERPILIH
# ============================================================

if "selected_foods" not in st.session_state:
    st.session_state.selected_foods = []

food_to_add = st.selectbox("Pilih makanan untuk ditambahkan:", filtered_df["name"].unique())

if st.button("➕ Tambahkan ke daftar konsumsi"):
    selected = filtered_df[filtered_df["name"] == food_to_add].iloc[0].to_dict()
    st.session_state.selected_foods.append(selected)
    st.success(f"✅ {food_to_add} berhasil ditambahkan!")

# ============================================================
# 🔹 TAMPILKAN DAFTAR MAKANAN
# ============================================================

if st.session_state.selected_foods:
    st.subheader("🧾 Daftar Makanan yang Dikonsumsi Hari Ini")

    selected_df = pd.DataFrame(st.session_state.selected_foods)
    st.dataframe(selected_df[["name", "calories", "fat", "proteins", "carbohydrate"]])

    # Hitung total nutrisi
    total_calories = selected_df["calories"].sum()
    total_fat = selected_df["fat"].sum()
    total_proteins = selected_df["proteins"].sum()
    total_carbs = selected_df["carbohydrate"].sum()

    st.markdown("### 🔢 Total Nutrisi Harian")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kalori (kkal)", f"{total_calories:.1f}")
    col2.metric("Lemak (g)", f"{total_fat:.1f}")
    col3.metric("Protein (g)", f"{total_proteins:.1f}")
    col4.metric("Karbohidrat (g)", f"{total_carbs:.1f}")

    # ============================================================
    # 🔮 PREDIKSI MODEL
    # ============================================================

    if st.button("🔮 Prediksi Skor Kesehatan"):
        # Bentuk data sesuai input model
        new_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Weight": weight,
            "Height": height,
            "Daily_Calories": total_calories,
            "Fat_Intake_g": total_fat,
            "Protein_Intake_g": total_proteins,
            "Sugar_Intake_g": total_carbs   # karena dataset pakai carbohydrate
        }])

        # Prediksi
        predicted_score = model.predict(new_data)[0]

        # Tentukan kategori
        if predicted_score >= 4:
            kategori = "🌿 Baik"
        elif predicted_score >= 2:
            kategori = "⚖️ Sedang"
        else:
            kategori = "💀 Buruk"

        # Tampilkan hasil
        st.success(f"**Prediksi Skor Kesehatan (1–5): {predicted_score}**")
        st.info(f"**Kategori Kesehatan: {kategori}**")

        st.progress(predicted_score / 5)

# ============================================================
# 🔹 FOOTER
# ============================================================

st.markdown("---")
st.caption("💡 Dibuat dengan Streamlit | Prediksi berdasarkan model Random Forest | Dataset: nutrition.csv")

