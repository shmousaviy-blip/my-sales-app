import pandas as pd
import streamlit as st
import io
import plotly.express as px
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq


# --- Header Section ---
st.set_page_config(page_title="Smart Data Analysis", page_icon="📈", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>🚀 هوش مصنوعی تحلیل فروش</h1>
    <p style='text-align: center; color: #888;'>فایل اکسل خود را آپلود کنید و تحلیل‌های پیشرفته بگیرید</p>
    <hr>
""", unsafe_allow_html=True)

# --- کد مربوط به تب‌ها و تحلیل‌ها که قبلاً داشتی اینجا قرار می‌گیرد ---

# --- Footer (همیشه در پایین صفحه) ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---")
footer_html = f"""
<div style="text-align: center;">
    <p style="margin-bottom: 10px;">👨‍💻 Developed By <b>اHassan Moosavi</b></p>
    <a href="https://wa.me/31685529172" target="_blank"><img src="https://img.shields.io/badge/WhatsApp-25D366?style=for-the-badge&logo=whatsapp&logoColor=white" style="margin:5px;"></a>
    <a href="http://linkedin.com/in/hassan-moosavi" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" style="margin:5px;"></a>
    <a href="mailto:s.h.mousaviy@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" style="margin:5px;"></a>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)








# --- Create an Excel template for importing the new titles ---
def create_template():
    template_data = {
        'Old_Column_Name': ['', ''],
        'New_Name': ['', '']
    }
    template_df = pd.DataFrame(template_data)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Sheet1')
    return buffer


# --- Sidebar Configuration ---
st.sidebar.subheader("AI Settings")
groq_api_key = st.sidebar.text_input("Enter Groq API Key:",
                                     value="",
                                     type="password",
                                     placeholder="gsk_...")

st.sidebar.subheader("Input Data")
data_file = st.sidebar.file_uploader("Choose a Sales Data", type="xlsx")

st.sidebar.subheader("Config The Titles Of Data")
needs_rename = st.sidebar.checkbox("Do you Need to Change the Data Titles?")

config_file = None
if needs_rename:
    template_file = create_template()
    st.sidebar.download_button(
        label="📥 Sample Package",
        data=template_file.getvalue(),
        file_name="Config_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    config_file = st.sidebar.file_uploader("Choose the Titles File", type="xlsx")

# --- Main Logic ---
if data_file:
    df = pd.read_excel(data_file)

    if needs_rename and config_file:
        config_df = pd.read_excel(config_file).dropna()
        name_map = dict(zip(config_df.iloc[:, 0], config_df.iloc[:, 1]))
        df = df.rename(columns=name_map)
        st.sidebar.success("columns title replaced successfully")
    elif needs_rename and not config_file:
        st.warning("please choose a file to Rename the title or unselect the title checkbox")
        st.stop()

    tab_stats, tab_charts, tab_bot = st.tabs(["📊 Stats", "📈 Analytics", "🤖 AI Bot"])

    # --- TAB 1: STATS ---
    with tab_stats:
        st.subheader("📊 Overview Of Data")
        desc_df = df.describe().T
        styled_desc = desc_df.style.format("{:,.0f}").set_properties(**{'text-align': 'right', 'padding': '10px'})
        with st.container():
            st.markdown('<div style="direction: rtl;">', unsafe_allow_html=True)
            st.dataframe(styled_desc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: ANALYTICS ---
    with tab_charts:
        st.subheader("📈 Analytics Report Settings")

        # اضافه شدن Pie Chart به لیست
        chart_type = st.radio("Choose Chart Type:", ["Line Chart", "Bar Chart", "Pie Chart", "Treemap"],
                              horizontal=True)

        with st.form("main_chart_form"):
            columns = df.columns.tolist()

            if chart_type == "Treemap":
                st.info("Hierarchical Settings")
                c1, c2 = st.columns(2)
                parent_level = c1.selectbox("Parent Category:", columns, key="parent_t")
                child_level = c2.selectbox("Child Category:", columns, key="child_t")
                y_axis = st.selectbox("Values (Numeric):", columns, key="val_t")
            else:
                st.info(f"{chart_type} Settings")
                c1, c2 = st.columns(2)
                x_axis = c1.selectbox("Select X axis / Category:", columns, key="x_val")
                y_axis = c2.selectbox("Select Y axis / Values:", columns, key="y_val")

            run_button = st.form_submit_button("🚀 Run Analysis")

        if run_button:
            st.divider()
            try:
                if chart_type == "Treemap":
                    chart_data = df.groupby([parent_level, child_level])[y_axis].sum().reset_index()
                    fig = px.treemap(chart_data, path=[parent_level, child_level], values=y_axis, color=y_axis,
                                     color_continuous_scale='RdBu')
                    fig.update_traces(textinfo="label+value")

                else:
                    chart_data = df.groupby(x_axis)[y_axis].sum().reset_index()
                    if chart_type == "Pie Chart":
                        fig = px.pie(chart_data, names=x_axis, values=y_axis, hole=0.3, template="plotly_white")
                    elif chart_type == "Line Chart":
                        chart_data[x_axis] = chart_data[x_axis].astype(str)
                        fig = px.line(chart_data, x=x_axis, y=y_axis, markers=True, template="plotly_white")
                    elif chart_type == "Bar Chart":
                        fig = px.bar(chart_data, x=x_axis, y=y_axis, template="plotly_white", color=y_axis)

                fig.update_layout(yaxis_tickformat=',.0f')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}. Check if you selected numeric columns for values.")

    # --- TAB 3: AI BOT ---
    with tab_bot:
        st.subheader("💬 Chat with your Data")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        if not groq_api_key:
            st.warning("⚠️ Please enter your Groq API Key in the sidebar Settings.")
        else:
            try:
                llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", api_key=groq_api_key)
                agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True,
                                                      handle_parsing_errors=True)

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

                if prompt := st.chat_input("Ask: Who are the top 5 customers?"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("🤖 Analyzing..."):
                            try:
                                full_prompt = f"Data columns: {list(df.columns)}. Task: {prompt}. ALWAYS include the result in your response. Answer in Persian."
                                response = agent.invoke({"input": full_prompt})
                                final_answer = response.get("output", str(response)) if isinstance(response,
                                                                                                   dict) else response
                                st.write(final_answer)
                                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                            except Exception as e:
                                st.error(f"AI Error: {e}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
else:
    st.info("Waiting for Sales Data to be uploaded...")

