import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="Hea",
    layout="wide",
    initial_sidebar_state="expanded"
)
def run():
    # Membuat Tittle
    st.title("Customer Churn Risk Prediction")

    # Membuat Sub Header
    st.subheader("EDA untuk Analisa Dataset churn risk")

    # Menambahkan Gambar
    image = Image.open('mls2.png')
    st.image(image)

    st.markdown("---")

    df = pd.read_csv("churn.csv")
    df = df.dropna()
    st.dataframe(df)

    target = ['churn_risk_score'] # target
    num_cols = ['days_since_last_login', 'avg_time_spent', 'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet'] # Numeric
    catn_cols = ['membership_category', 'feedback']

    st.markdown("---")
    st.write("## 1. Eksplorasi Kolom Target")
    target = df["churn_risk_score"].value_counts().reset_index()
    persen = df["churn_risk_score"].value_counts(normalize=True).reset_index()
    target["percentage"] = persen["churn_risk_score"]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.pie(target["percentage"], labels = target["index"], autopct='%.0f%%')
    ax.set_title('Pie Chart churn risk score')
    ax.legend(fontsize=12)
    st.pyplot(fig)
    st.dataframe(target)
    st.write("Dari data dan plot diatas didapatkan bahwa distribusi data pada kolom target atau dependen variable itu balance. Hal ini karena perbandingan rasio antara customer dengan resiko churn dan yang tidak beresiko churn itu 54:46.")
    st.markdown("---")

    st.write("## 2. Eksplorasi Kolom Categorical")
    def autopct_format(values):
            def my_format(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return '{:.1f}%\n({v:d})'.format(pct, v=val)
            return my_format
    df_0 = df[df['churn_risk_score'] == 0]
    df_1 = df[df['churn_risk_score'] == 1]

    st.write("### 2.1. Eksplorasi Kolom membership_category")
    plt.figure(figsize=(20,5))
    sns.countplot(df["membership_category"], hue = df["churn_risk_score"])
    plt.title("Klasifikasi Churn Risk Customer berdasarkan kolom membership_category")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    st.pyplot(plt)
    plt.figure(figsize=(10,5))
    s = df['membership_category'].value_counts()
    plt.pie(s, labels=s.index, autopct=autopct_format(s))
    plt.title("Pie Chart Kolom membership_category")
    st.pyplot(plt)
    plt.figure(figsize=(10,5))
    s = df_0['membership_category'].value_counts()
    plt.pie(s,labels = s.index, autopct=autopct_format(s))
    plt.title("Pie Chart Kolom membership_category yang memiliki churn risk 0")
    st.pyplot(plt)
    plt.figure(figsize=(10,5))
    s = df_1['membership_category'].value_counts()
    plt.pie(s,labels = s.index, autopct=autopct_format(s))
    plt.title("Pie Chart Kolom membership_category yang memiliki churn risk 1")
    st.pyplot(plt)
    st.write('Setelah dilakukan Eksplorasi didapatkan bahwa data pada kolom ini berpengaruh terhadap penentuan nilai churn risk, karena distribusi data yang memiliki nilai churn risk 0 sangatlah berbeda dengan distribusi data yang memiliki nilai churn risk 1. Oleh karena itu kolom ini akan digunakan dalam permodelan. Selain itu dari data dan plot diatas juga didapatkan bahwa customer dengan churn risk terendah adalah customer dengan category premium atau platinum membership. Hal ini dapat dibuktikan pada plot bahwa customer yang berlangganan premium atau platinum membership tidak ada yang memiliki churn risk.')

    st.write("### 2.2. Eksplorasi Kolom feedback")
    plt.figure(figsize=(30,5))
    sns.countplot(df["feedback"], hue = df["churn_risk_score"])
    plt.title("Klasifikasi Churn Risk Customer berdasarkan kolom feedback")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    st.pyplot(plt)
    plt.figure(figsize=(10,5))
    s = df['feedback'].value_counts()
    plt.pie(s, labels=s.index, autopct=autopct_format(s))
    plt.title("Pie Chart Kolom feedback")
    st.pyplot(plt)
    plt.figure(figsize=(10,5))
    s = df_0['feedback'].value_counts()
    plt.pie(s,labels = s.index, autopct=autopct_format(s))
    plt.title("Pie Chart Kolom feedback yang memiliki churn risk 0")
    st.pyplot(plt)
    plt.figure(figsize=(10,5))
    s = df_1['feedback'].value_counts()
    plt.pie(s,labels = s.index, autopct=autopct_format(s))
    plt.title("Pie Chart Kolom feedback yang memiliki churn risk 1")
    st.pyplot(plt)
    st.write('Setelah dilakukan Eksplorasi didapatkan bahwa data pada kolom ini berpengaruh terhadap penentuan nilai churn risk, karena distribusi data yang memiliki nilai churn risk 0 sangatlah berbeda dengan distribusi data yang memiliki nilai churn risk 1. Oleh karena itu kolom ini akan digunakan dalam permodelan. Selain itu kita juga mendapatkan informasi bahwa customer yang meninggalkan feedback positif tidak memiliki resiko untuk churn. Sedangkan customer yang meninggalkan feedback negatif atau netral memiliki resiko untuk churn.')

    st.write("### 2.3. Eksplorasi Kolom gender")
    plt.figure(figsize=(7,5))
    sns.countplot(df["gender"], hue = df["churn_risk_score"])
    plt.title("Klasifikasi Churn Risk Customer berdasarkan kolom Gender")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    st.pyplot(plt)
    plt.figure(figsize=(10,5))
    s = df['gender'].value_counts()
    plt.pie(s,labels = s.index, autopct=autopct_format(s))
    plt.title("Pie Chart Kolom Gender")
    st.pyplot(plt)
    plt.figure(figsize=(10,5))
    s = df_0['gender'].value_counts()
    plt.pie(s,labels = s.index, autopct=autopct_format(s))
    plt.title("Pie Chart Kolom Gender yang memiliki churn risk 0")
    st.pyplot(plt)
    plt.figure(figsize=(10,5))
    s = df_1['gender'].value_counts()
    plt.pie(s,labels = s.index, autopct=autopct_format(s))
    plt.title("Pie Chart Kolom Gender yang memiliki churn risk 1")
    st.pyplot(plt)
    st.write('Karena pada kolom ini nilai distribusi data sangat seimbang dan distribusi data pada kolom ini sangatlah mirip saat data di group berdasarkan jenis churn risk yang sama. Maka data akan di drop karena data pada kolom ini tidak memiliki pengaruh besar terhadap penentuan nilai churn risk. Selain itu dari plot juga kita dapatkan bahwa lebih banyak customer female ddibanding male. akan tetapi customer Male memiliki tingkat churn risk yang lebih rendah dari female.')

    st.write("## 3. Eksplorasi Kolom Numerical")
    st.write("### 3.1. Eksplorasi kolom avg_time_spent")
    plt.figure(figsize=(12,5))
    sns.histplot(df["avg_time_spent"])
    plt.title("avg_time_spent")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    st.pyplot(plt)
    no_churn = df[df["churn_risk_score"] == 0]["avg_time_spent"].reset_index()
    churn = df[df["churn_risk_score"] == 1]["avg_time_spent"].reset_index()

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(13,5))
    axes[0].hist(no_churn["avg_time_spent"])
    axes[0].set_title("avg_time_spent customer yang tidak beresiko churn")
    axes[1].hist(churn["avg_time_spent"], color='orange')
    axes[1].set_title("avg_time_spent customer yang beresiko churn")

    st.pyplot(plt)
    no_churn_desc = no_churn.describe()
    churn_desc = churn.describe()
    st.write('Describe customer with no risk of churn in column avg_time_spent')
    st.dataframe(no_churn_desc)
    st.write('Describe customer with risk of churn in column avg_time_spent')
    st.dataframe(churn_desc)
    st.write("Dari data dan plot diatas didapatkan bahwa rata-rata column avg_time_spent yang beresiko churn itu lebih rendah dari rata-rata column avg_time_spent yang tidak beresiko churn.")


    st.write("### 3.2. Eksplorasi kolom avg_transaction_value")
    plt.figure(figsize=(12,5))
    sns.histplot(df["avg_transaction_value"])
    plt.title("avg_transaction_value")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    st.pyplot(plt)
    no_churn = df[df["churn_risk_score"] == 0]["avg_transaction_value"].reset_index()
    churn = df[df["churn_risk_score"] == 1]["avg_transaction_value"].reset_index()

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(13,5))
    axes[0].hist(no_churn["avg_transaction_value"])
    axes[0].set_title("avg_transaction_value customer yang tidak beresiko churn")
    axes[1].hist(churn["avg_transaction_value"], color='orange')
    axes[1].set_title("avg_transaction_value customer yang beresiko churn")

    st.pyplot(plt)
    no_churn_desc = no_churn.describe()
    churn_desc = churn.describe()
    st.write('Describe customer with no risk of churn in column avg_transaction_value')
    st.dataframe(no_churn_desc)
    st.write('Describe customer with risk of churn in column avg_transaction_value')
    st.dataframe(churn_desc)
    st.write("Dari data dan plot diatas didapatkan bahwa nilai rata-rata column avg_transaction_value yang beresiko churn itu lebih rendah dari nilai rata-rata column avg_transaction_value yang tidak beresiko churn.")

    st.write("### 3.3. Eksplorasi kolom points_in_wallet")
    plt.figure(figsize=(12,5))
    sns.histplot(df["points_in_wallet"])
    plt.title("points_in_wallet")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    st.pyplot(plt)
    no_churn = df[df["churn_risk_score"] == 0]["points_in_wallet"].reset_index()
    churn = df[df["churn_risk_score"] == 1]["points_in_wallet"].reset_index()

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(13,5))
    axes[0].hist(no_churn["points_in_wallet"])
    axes[0].set_title("points_in_wallet customer yang tidak beresiko churn")
    axes[1].hist(churn["points_in_wallet"], color='orange')
    axes[1].set_title("points_in_wallet customer yang beresiko churn")

    st.pyplot(plt)
    no_churn_desc = no_churn.describe()
    churn_desc = churn.describe()
    st.write('Describe customer with no risk of churn in column points_in_wallet')
    st.dataframe(no_churn_desc)
    st.write('Describe customer with risk of churn in column points_in_wallet')
    st.dataframe(churn_desc)
    st.write("Dari data dan plot diatas didapatkan bahwa customer yang beresiko churn itu memiliki nilai points_in_wallet yang lebih rendah dari customer yang tidak beresiko untuk churn.")

if __name__ == "__app__":
    run()