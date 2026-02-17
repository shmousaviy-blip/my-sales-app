import pandas as pd
import streamlit as st
import io
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

# --- 1. تنظیمات اولیه صفحه ---
st.set_page_config(page_title="Data Analysis Assistant", page_icon="📈", layout="wide")


# --- 2. توابع کمکی ---
def create_template():
    template_data = {'Old_Column_Name': ['', ''], 'New_Name': ['', '']}
    template_df = pd.DataFrame(template_data)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Sheet1')
    return buffer


def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast_Results')
    return output.getvalue()


# --- 3. هدر اصلی برنامه ---
st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: #4A90E2; margin: 0;'>🚀 Data Analysis Assistant</h1>
        <p style='color: #666; font-size: 1.1rem;'>Smart, fast and accurate for analyzing your data</p>
    </div>
""", unsafe_allow_html=True)

# --- 4. تنظیمات سایدبار ---
st.sidebar.title("🛠️ Control Panel")
st.sidebar.divider()
st.sidebar.subheader("🔑 AI Settings")
groq_api_key = st.sidebar.text_input("Enter Groq API Key:", value="", type="password", placeholder="gsk_...")
st.sidebar.divider()
st.sidebar.subheader("📂 Input Data")
data_file = st.sidebar.file_uploader("Choose a Sales Data", type="xlsx")
st.sidebar.divider()
st.sidebar.subheader("📝 Config The Titles Of Data")
needs_rename = st.sidebar.checkbox("Do you need to change the data titles?")

config_file = None
if needs_rename:
    template_file = create_template()
    st.sidebar.download_button(label="📥 Sample Package", data=template_file.getvalue(),
                               file_name="Config_Template.xlsx")
    config_file = st.sidebar.file_uploader("Choose the titles file", type="xlsx")

# تعریف Footer
footer_html = f"""
<div style="text-align: center;">
    <p style="margin-bottom: 10px; font-size: 0.9rem; color: #555;">Developed by <b>Hassan Moosavi</b></p>
    <div style="display: flex; justify-content: center; gap: 15px;">
        <a href="https://wa.me/31685529172" target="_blank"><img src="https://img.shields.io/badge/WhatsApp-25D366?style=flat-square&logo=whatsapp&logoColor=white" height="25"></a>
        <a href="http://linkedin.com/in/hassan-moosavi" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white" height="25"></a>
        <a href="mailto:s.h.mousaviy@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white" height="25"></a>
    </div>
</div>
"""

# --- 5. منطق اصلی برنامه ---
if data_file:
    df = pd.read_excel(data_file)

    if needs_rename and config_file:
        config_df = pd.read_excel(config_file).dropna()
        name_map = dict(zip(config_df.iloc[:, 0], config_df.iloc[:, 1]))
        df = df.rename(columns=name_map)
        st.sidebar.success("✅ Columns titles replaced!")
    elif needs_rename and not config_file:
        st.warning("Please choose a file to Rename the titles or unselect the checkbox.")
        st.stop()

    tab_stats, tab_charts, tab_bot = st.tabs(["📊 Stats", "📈 Analytics", "🤖 AI Bot"])

    with tab_stats:
        st.subheader("📊 Overview Of Data")
        st.dataframe(df.describe(include=[np.number]).T.style.format("{:,.0f}"), use_container_width=True)

        st.divider()
        st.subheader("🔮 Aggregated Forecasting (Sum of Value per Time)")

        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        c_select1, c_select2, c_select3 = st.columns(3)
        time_col = c_select1.selectbox("📅 Select Time/Date Column:", ["-- Choose --"] + cols)
        value_col = c_select2.selectbox("💰 Select Value Column to Predict:", ["-- Choose --"] + numeric_cols)
        period_type = c_select3.selectbox("⏱️ Forecast Basis:", ["Daily", "Monthly"])

        if time_col != "-- Choose --" and value_col != "-- Choose --":
            try:
                temp_df = df.dropna(subset=[time_col, value_col]).copy()
                if period_type == "Monthly":
                    temp_df['Grouping_Time'] = temp_df[time_col].astype(str).str[:7]
                else:
                    temp_df['Grouping_Time'] = temp_df[time_col].astype(str)

                agg_df = temp_df.groupby('Grouping_Time')[value_col].sum().reset_index()
                agg_df = agg_df.sort_values(by='Grouping_Time')

                y = agg_df[value_col].values.reshape(-1, 1)
                X = np.arange(len(y)).reshape(-1, 1)

                if len(y) > 2:
                    future_steps = st.slider(f"Forecast {period_type} periods into future:", 1, 30, 5)
                    model = LinearRegression().fit(X, y)
                    X_future = np.arange(len(y), len(y) + future_steps).reshape(-1, 1)
                    y_pred = model.predict(X_future)

                    st.info(f"💡 Analysis performed on the **Total Sum** of `{value_col}` per each `{period_type}`.")

                    st.write(f"**Predicted Total {value_col}:**")
                    pred_results = pd.DataFrame({
                        'Period': [f"Future +{i + 1}" for i in range(future_steps)],
                        'Predicted Total': y_pred.flatten()
                    })
                    st.dataframe(pred_results.style.format(subset=['Predicted Total'], formatter="{:,.2f}"),
                                 use_container_width=True)

                    st.download_button(label="📥 Download Forecast Report", data=to_excel(pred_results),
                                       file_name="Forecast_Report.xlsx")

                    history_df = pd.DataFrame({'Time': agg_df['Grouping_Time'].astype(str), 'Value': agg_df[value_col],
                                               'Type': 'Actual Total'})
                    future_df = pd.DataFrame(
                        {'Time': [f"Next {i + 1}" for i in range(future_steps)], 'Value': y_pred.flatten(),
                         'Type': 'Forecast'})

                    fig_forecast = px.line(pd.concat([history_df, future_df]), x='Time', y='Value', color='Type',
                                           title=f"Trend of Total {value_col} per {period_type}",
                                           markers=True,
                                           color_discrete_map={'Actual Total': '#4A90E2', 'Forecast': '#2ECC71'})
                    fig_forecast.update_layout(yaxis_tickformat=',.0f', template="plotly_white")
                    st.plotly_chart(fig_forecast, use_container_width=True)
                else:
                    st.info("ℹ️ Not enough unique periods after aggregation for forecasting.")
            except Exception as e:
                st.error(f"⚠️ Calculation Error: {e}")
        else:
            st.info("👋 Please select both Time and Value columns to see the aggregated forecast.")

    with tab_charts:
        st.subheader("📈 Analytics Report Settings")
        chart_type = st.radio("Choose Chart Type:", ["Line Chart", "Bar Chart", "Pie Chart", "Treemap"],
                              horizontal=True)
        with st.expander("⚙️ Configure Chart Axes", expanded=True):
            with st.form("main_chart_form"):
                columns = df.columns.tolist()
                if chart_type == "Treemap":
                    c1, c2, c3 = st.columns(3)
                    parent_level = c1.selectbox("Parent Category:", columns)
                    child_level = c2.selectbox("Child Category:", columns)
                    y_axis = c3.selectbox("Values (Numeric):", columns)
                else:
                    c1, c2 = st.columns(2)
                    x_axis = c1.selectbox("Select X axis:", columns)
                    y_axis = c2.selectbox("Select Y axis:", columns)
                run_button = st.form_submit_button("🚀 Run Analysis")

        if run_button:
            try:
                chart_data = df.groupby([parent_level, child_level] if chart_type == "Treemap" else x_axis)[
                    y_axis].sum().reset_index()
                if chart_type == "Treemap":
                    fig = px.treemap(chart_data, path=[parent_level, child_level], values=y_axis, color=y_axis,
                                     color_continuous_scale='RdBu')
                elif chart_type == "Pie Chart":
                    fig = px.pie(chart_data, names=x_axis, values=y_axis, hole=0.3)
                elif chart_type == "Line Chart":
                    fig = px.line(chart_data, x=x_axis, y=y_axis, markers=True)
                elif chart_type == "Bar Chart":
                    fig = px.bar(chart_data, x=x_axis, y=y_axis, color=y_axis)
                fig.update_layout(yaxis_tickformat=',.0f', template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    with tab_bot:
        st.subheader("💬 Chat with your Data")
        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
            st.rerun()
        if not groq_api_key:
            st.warning("⚠️ Please enter your Groq API Key in the sidebar Settings.")
        else:
            try:
                llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", api_key=groq_api_key)
                # allow_dangerous_code set to True for data processing speed, but UI chart logic removed.
                agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True,
                                                      handle_parsing_errors=True)

                if "messages" not in st.session_state: st.session_state.messages = []
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]): st.write(msg["content"])

                if prompt := st.chat_input("Ask about your data..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.write(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("🤖 Thinking..."):
                            # Logic: No Persian responses, strictly English. No plotting instructions.
                            response = agent.invoke({"input": f"Respond strictly in English. Question: {prompt}"})
                            final_answer = response.get("output", str(response)) if isinstance(response,
                                                                                               dict) else response
                            st.write(final_answer)
                            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            except Exception as e:
                st.error(f"AI Error: {e}")
else:
    st.markdown("""
        <div style='text-align: center; padding: 5rem; border: 2px dashed #ccc; border-radius: 20px; color: #888;'>
            <img src='https://cdn-icons-png.flaticon.com/512/4090/4090458.png' width='100' style='opacity: 0.5; margin-bottom: 1rem;'>
            <h3>Waiting for Data to be uploaded from the sidebar...</h3>
            <p>Please upload your Excel file from the left menu to begin the analysis.</p>
        </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown(footer_html, unsafe_allow_html=True)
