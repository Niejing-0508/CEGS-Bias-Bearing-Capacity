import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import streamlit as st

# --- 页面配置 ---
st.set_page_config(
    page_title="CEGS Bias bearing capacity prediction system",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 全局设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 全局变量 ---
feature_order = ['r', 'L', 'λ', 'e', 'θ', 'D', 't', 'fc', 'fy', 'α', 'F', 'SF', 'Sα']


# --- 函数定义 ---

# 加载模型的函数
@st.cache_resource
def load_model():
    """加载训练好的模型和特征缩放器"""
    try:
        with open("CEGS_optuna_ensemble_model.pkl", "rb") as f:
            saved_data = pickle.load(f)
        model = saved_data["ensemble_model"]
        scaler = saved_data["feature_scaler"]
        st.success("The model has been loaded successfully！")
        return model, scaler
    except FileNotFoundError:
        st.error("The model file CEGS_optuna_ensemble_model.pkl was not found. Please ensure that the model file is in "
                 "the same directory as the application.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model：{str(e)}")
        return None, None


# 验证输入的函数
def validate_inputs(inputs):
    """验证输入是否有效"""
    try:
        validated = {}
        for key, value in inputs.items():
            if value is None:  # Streamlit的number_input为空时返回None
                raise ValueError(f"{key} Parameters cannot be empty")
            validated[key] = float(value)
        # 检查参数范围
        if not (0 <= validated['r'] <= 1): raise ValueError("r should be between 0 and 1")
        if not (100 <= validated['L'] <= 10000): raise ValueError("The L (mm) should be between 100 and 10,000")
        if not (0 <= validated['λ'] <= 70): raise ValueError("λ should be between 0 and 70")
        if not (0 <= validated['e'] <= 200): raise ValueError("e (mm) should be between 0 and 200")
        if not (0 <= validated['θ'] <= 5): raise ValueError("θ should be between 0 and 5")
        if not (50 <= validated['D'] <= 300): raise ValueError("The D/B (mm) ratio should be between 50 and 300")
        if not (0 <= validated['t'] <= 10): raise ValueError("t (mm) should be between 0 and 10")
        if not (0 <= validated['fc'] <= 80): raise ValueError("The fc (MPa) should be between 0 and 80")
        if not (0 <= validated['fy'] <= 1200): raise ValueError("The fy (MPa) should be between 0 and 1200")
        if not (0 <= validated['α'] <= 0.5): raise ValueError("α should be between 0 and 0.5")
        if not (1 <= validated['F'] <= 2): raise ValueError("F should be between 1 and 2")
        if not (0 <= validated['SF'] <= 2): raise ValueError("SF should be between 0 and 2")
        if not (0 <= validated['Sα'] <= 0.5): raise ValueError("Sα should be between 0 and 0.5")

        return validated
    except ValueError as e:
        st.warning(str(e))
        return None


# --- 主程序 ---
def main():
    # 初始化session_state中的清空标志
    if 'clear_inputs' not in st.session_state:
        st.session_state.clear_inputs = False

    st.title("🏗️ CEGS Bias bearing capacity prediction system")

    # 加载模型
    model, scaler = load_model()

    # 创建标签页
    tab1, tab2 = st.tabs(["Single-sample prediction", "Batch prediction"])

    # --- 单样本预测标签页 ---
    with tab1:
        st.header("Single-sample prediction")
        st.markdown("Please enter the following parameters for prediction：")

        # ********** 修改为三列布局 **********
        col1, col2, col3 = st.columns(3)

        # 参数信息列表
        params_info = [
            ('r', 'The replacement rate of recycled concrete [r] ', 0.5, 0.0, 1.0),
            ('L', 'Column height [L(mm)]', 1000.0, 100.0, 10000.0),
            ('λ', 'Slenderness ratio [λ]', 10.0, 0.0, 70.0),
            ('e', 'Offset [e(mm)]', 50.0, 0.0, 200.0),
            ('θ', 'Confining factor [θ]', 1.0, 0.0, 5.0),
            ('D', 'Side length/Diameter [D/B(mm)]', 150.0, 50.0, 300.0),
            ('t', 'Thickness of steel pipe [t(mm)]', 5.0, 0.0, 10.0),
            ('fc', 'Compressive strength of core concrete [fc(MPa)]', 40.0, 0.0, 80.0),
            ('fy', 'Yield strength of steel pipes [fy(MPa)]', 400.0, 0.0, 1200.0),
            ('α', 'Steel ratio [α]', 0.02, 0.0, 0.5),
            ('F', 'Column section form [F]', 1.0, 1.0, 2.0),
            ('SF', 'Steel section form [SF]', 1.0, 0.0, 2.0),
            ('Sα', 'Steel content of section steel [Sα]', 0.01, 0.0, 0.5),
        ]

        user_inputs = {}

        # 循环创建输入框 (每行三列)
        for i, (key, label, default_value, min_val, max_val) in enumerate(params_info):
            # 关键逻辑：根据 st.session_state.clear_inputs 决定输入框的值
            current_value = None if st.session_state.clear_inputs else default_value

            if i % 3 == 0:
                with col1:
                    user_inputs[key] = st.number_input(
                        label,
                        value=current_value,
                        min_value=min_val,
                        max_value=max_val,
                        step=0.01 if key in ['r', 'α', 'Sα', 'θ'] else 1.0,
                        format="%.4f" if key in ['r', 'α', 'Sα'] else "%.1f"
                    )
            elif i % 3 == 1:
                with col2:
                    user_inputs[key] = st.number_input(
                        label,
                        value=current_value,
                        min_value=min_val,
                        max_value=max_val,
                        step=0.01 if key in ['r', 'α', 'Sα', 'θ'] else 1.0,
                        format="%.4f" if key in ['r', 'α', 'Sα'] else "%.1f"
                    )
            else:
                with col3:
                    user_inputs[key] = st.number_input(
                        label,
                        value=current_value,
                        min_value=min_val,
                        max_value=max_val,
                        step=0.01 if key in ['r', 'α', 'Sα', 'θ'] else 1.0,
                        format="%.4f" if key in ['r', 'α', 'Sα'] else "%.1f"
                    )

        # 按钮布局
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 8])

        with btn_col1:
            predict_button = st.button("Start the prediction", type="primary")

        with btn_col2:
            clear_button = st.button("Clear Inputs")

        # 处理清空按钮逻辑
        if clear_button:
            st.session_state.clear_inputs = True
            st.rerun()  # 重新运行以清空输入框

        # 处理预测按钮逻辑
        if predict_button:
            st.session_state.clear_inputs = False  # 预测后恢复默认状态

            if not model or not scaler:
                st.warning("The model has not been loaded and thus cannot make predictions.")
                return

            validated_inputs = validate_inputs(user_inputs)
            if not validated_inputs:
                return

            try:
                # 准备特征数据
                features = np.array([validated_inputs[feat] for feat in feature_order]).reshape(1, -1)
                scaled_features = scaler.transform(features)

                # 进行预测
                Nu = model.predict(scaled_features)[0]
                individual_preds = model.predict_individual(scaled_features)

                # 显示结果
                st.subheader("Outcome")
                result_container = st.container(border=True)
                with result_container:
                    st.write(f"**Integrated model prediction of Nu (Bias bearing Capacity):** {Nu:.2f} kN")

                    st.write("**The prediction results of each sub-model:**")
                    for model_name, pred_value in individual_preds.items():
                        st.write(f"  - {model_name}: {pred_value[0]:.2f} kN")

                    st.write("**Model weight:**")
                    for model_name, weight in model.weights.items():
                        st.write(f"  - {model_name}: {weight:.4f} ({weight * 100:.1f}%)")

            except Exception as e:
                st.error(f"Error in prediction：{str(e)}")

    # --- 批量预测标签页 ---
    with tab2:
        st.header("Batch prediction")
        st.markdown("Please upload an Excel file containing the required parameters for batch prediction.")

        # 文件上传
        uploaded_file = st.file_uploader("Select the Excel file", type=["xlsx", "xls"])

        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                st.success(f"The file has been uploaded successfully! A total of {len(df)} data points were detected.")

                # 数据预览
                st.subheader("Data preview")
                st.dataframe(df.head())

                # 检查必要列是否存在
                missing_cols = [col for col in feature_order if col not in df.columns]
                if missing_cols:
                    st.warning(f"The uploaded file is missing the necessary columns：{', '.join(missing_cols)}")
                else:
                    # 开始预测按钮
                    if st.button("Start batch prediction", type="primary"):
                        if not model or not scaler:
                            st.warning("The model has not been loaded and thus cannot make predictions.")
                            return

                        with st.spinner("Batch prediction is underway. Please wait a moment..."):
                            # 特征准备与预测
                            features = df[feature_order].values
                            scaled_features = scaler.transform(features)
                            batch_Nu = model.predict(scaled_features)

                            # 生成结果
                            result_df = df.copy()
                            result_df['Integrated model prediction Nu (kN)'] = batch_Nu

                            # 如果包含真实值，计算误差
                            if 'Nu' in df.columns:
                                result_df['Absolute error (kN)'] = np.abs(df['Nu'] - batch_Nu)
                                result_df['Relative error (%)'] = np.abs((df['Nu'] - batch_Nu) / df['Nu']) * 100

                                # 计算统计指标
                                avg_abs_error = np.mean(result_df['Absolute error (kN)'])
                                avg_rel_error = np.mean(result_df['Relative error (%))'])

                                st.subheader("Statistics of prediction results")
                                stats_container = st.container(border=True)
                                with stats_container:
                                    st.write(f"**Mean absolute error (MAE):** {avg_abs_error:.2f} kN")
                                    st.write(f"**Average relative error:** {avg_rel_error:.2f}%")

                        # 显示结果并提供下载
                        st.subheader("Batch prediction results")
                        st.dataframe(result_df)

                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download the prediction results (CSV)",
                            data=csv,
                            file_name="Batch prediction results.csv",
                            mime="text/csv",
                        )

            except Exception as e:
                st.error(f"An error occurred while processing the file：{str(e)}")


# --- 运行应用 ---
if __name__ == "__main__":
    main()