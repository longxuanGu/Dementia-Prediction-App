import streamlit as st
import shap
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io  # 新增：用于 HTML 缓冲

# ========================
# Global Config (SCI Style)
# ========================
st.set_page_config(page_title="Dementia Stage Prediction", layout="centered")

# Times New Roman + 高清输出
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

# ========================
# Load Model
# ========================
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ========================
# Title & Sidebar Inputs
# ========================
st.title("Dementia Stage Prediction")

st.sidebar.header("Input Features")
education_years = st.sidebar.slider("Years of Education", 0, 30, 12, 1)
mmse = st.sidebar.slider("MMSE", 0, 30, 25, 1)
fdg_suvr = st.sidebar.slider("FDG_SUVR_Score", -2.00, 2.00, 0.00, 0.01)
amyloid_suvr = st.sidebar.slider("Amyloid_SUVR_Score", -2.00, 2.00, 0.00, 0.01)

# ========================
# Construct Input Data
# ========================
input_data = pd.DataFrame({
    "Education": [education_years],
    "MMSE": [mmse],
    "FDG_SUVR_Score": [fdg_suvr],
    "Amyloid_SUVR_Score": [amyloid_suvr]
}).reindex(columns=["Education", "MMSE", "FDG_SUVR_Score", "Amyloid_SUVR_Score"])

# ========================
# Prediction + SHAP
# ========================
if st.button("Run Prediction", type="primary", use_container_width=True):
    with st.spinner("Generating prediction and SHAP explanations..."):
        try:
            # ---------- Prediction ----------
            pred_class = int(model.predict(input_data)[0])
            pred_prob = model.predict_proba(input_data)[0][pred_class]

            class_map = {
                0: "Moderate to Severe Dementia",
                1: "Very Mild to Mild Dementia"
            }
            st.success(f"**Predicted Class:** {class_map[pred_class]}")
            st.write(f"**Confidence:** {pred_prob:.3f}")

            # ---------- SHAP Calculation ----------
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_data)

            if shap_values.values.ndim == 3:  # multiclass
                sv = shap_values.values[0, :, pred_class]
                base = explainer.expected_value[pred_class]
            else:
                sv = shap_values.values[0]
                base = explainer.expected_value

            explanation = shap.Explanation(
                values=sv,
                base_values=base,
                data=input_data.iloc[0],
                feature_names=input_data.columns
            )

            # ========================
            # Panel A — SHAP Force Plot（HTML 交互版，永不空白！）
            # ========================
            st.markdown("### **A. SHAP Force Plot**")
            # 生成 HTML（默认 JS 模式，稳定显示）
            force_plot_html = shap.plots.force(
                explanation,
                matplotlib=False,  # 默认 JS/HTML 模式
                show=False
            )
            shap.save_html("temp_force_plot.html", force_plot_html)  # 保存到临时文件
            with open("temp_force_plot.html", "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=300, scrolling=True)
            # 清理临时文件（Streamlit 会自动处理）
            import os
            if os.path.exists("temp_force_plot.html"):
                os.remove("temp_force_plot.html")

            st.caption(
                "**A.** SHAP force plot. "
                "Red = pushes toward mild dementia; Blue = pushes toward moderate-to-severe dementia."
            )

            # ========================
            # Panel B — SHAP Waterfall Plot（matplotlib 静态版）
            # ========================
            st.markdown("### **B. SHAP Waterfall Plot**")
            figB = plt.figure(figsize=(8, 5.5))
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(figB)
            plt.close(figB)

            st.caption(
                "**B.** Cumulative feature contributions from the base value (average prediction) "
                "to the final model output."
            )

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

# ========================
# Footer
# ========================
st.markdown("---")
st.caption("Dementia Stage Prediction App © 2025 | Powered by XGBoost + SHAP")