import streamlit as st
from streamlit_chat import message
from model import ChatGLMModel, chat_template


# page state

@st.cache_resource
def create_model():
    return ChatGLMModel()

with st.spinner("加载模型中..."):
    model = create_model()


if "history" not in st.session_state:
    st.session_state["history"] = []


# parameters

with st.sidebar:
    st.markdown("## 采样参数")

    max_tokens = st.number_input("max_tokens", min_value=1, max_value=500, value=200)
    temperature = st.number_input("temperature", min_value=0.1, max_value=4.0, value=1.0)
    top_p = st.number_input("top_p", min_value=0.1, max_value=1.0, value=0.7)
    top_k = st.number_input("top_k", min_value=1, max_value=500, value=50)

    if st.button("清空上下文"):
        st.session_state.message = ""
        st.session_state.history = []

    st.markdown("""
    [ChatGLM](https://huggingface.co/THUDM/chatglm-6b) + [ONNXRuntime](https://onnxruntime.ai/)
    """)


# main body

st.markdown("## ChatGLM + ONNXRuntime")

history: list[tuple[str, str]] = st.session_state.history

if len(history) == 0:
    st.caption("请在下方输入消息开始会话")


for idx, (question, answer) in enumerate(history):
    message(question, is_user=True, key=f"history_question_{idx}")
    st.write(answer)
    st.markdown("---")


next_answer = st.container()

question = st.text_area(label="消息", key="message")

if st.button("发送") and len(question.strip()):
    with next_answer:
        message(question, is_user=True, key="message_question")
        with st.spinner("正在回复中"):
            with st.empty():
                prompt = chat_template(history, question)
                for answer in model.generate_iterate(
                    prompt,
                    max_generated_tokens=max_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                ):
                    st.write(answer)
                message(answer, key="message_answer")

    st.session_state.history = history + [(question, answer)]
