import streamlit as st
import os
import glob
import numpy as np
import google.generativeai as genai
from pypdf import PdfReader
import time

st.set_page_config(page_title="AI Books Chatbot", layout="wide")
st.title("📚 نظام الدردشة مع الكتب الذكي")

# إعداد API Key من خلال Secrets في الموقع
api_key = st.sidebar.text_input("Google API Key", type="password")
if api_key:
    genai.configure(api_key=api_key)

    # وظائف معالجة الملفات
    def process_pdfs(files):
        chunks = []
        for pdf in files:
            reader = PdfReader(pdf)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    chunks.append(f'[كتاب: {pdf.name} - ص: {i+1}]\n{text}')
        
        res = genai.embed_content(model="models/embedding-001", content=chunks, task_type="retrieval_document")
        return {"chunks": chunks, "embeddings": np.array(res['embedding'])}

    uploaded_files = st.file_uploader("ارفع ملفات PDF الخاصة بك", accept_multiple_files=True, type="pdf")

    if uploaded_files:
        if 'db' not in st.session_state:
            with st.spinner("جاري معالجة الكتب وإنشاء قاعدة البيانات..."):
                st.session_state.db = process_pdfs(uploaded_files)
                st.success("✅ جاهز للدردشة!")

        query = st.text_input("اسأل أي سؤال حول الكتب:")
        if query and 'db' in st.session_state:
            with st.spinner("جاري البحث عن الإجابة..."):
                db = st.session_state.db
                q_emb = genai.embed_content(model="models/embedding-001", content=query, task_type="retrieval_query")['embedding']
                scores = np.dot(db["embeddings"], q_emb)
                top_idx = np.argsort(scores)[-5:][::-1]
                context = "\n---\n".join([db["chunks"][i] for i in top_idx])

                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                response = model.generate_content(f"أجب بدقة من النصوص:\n{context}\n\nالسؤال: {query}")
                st.markdown(f"### 🤖 الإجابة:\n{response.text}")
else:
    st.warning("يرجى إدخال مفتاح API Key في القائمة الجانبية للبدء.")
