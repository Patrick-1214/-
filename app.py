import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

st.title("手寫數字辨識 - 驗證展示")
st.write("這個網頁展示了一個使用 SVM 的手寫數字辨識模型，並顯示一張測試圖像與其預測結果。")

# 載入資料
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5)

# 訓練模型
clf = svm.SVC(gamma=0.001)
clf.fit(X_train, y_train)

# 預測第一筆資料
prediction = clf.predict([X_test[0]])
st.write(f"模型預測的數字是：**{prediction[0]}**")

# 顯示圖像
fig, ax = plt.subplots()
ax.imshow(X_test[0].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
st.pyplot(fig)