import streamlit as st
import pandas as pd
import json
import re
import openai

# ==================== 工具函数 ====================

def read_csv_with_encoding(file):
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-16', 'latin1']
    for encoding in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except:
            continue
    raise ValueError("无法读取 CSV 文件")


def detect_languages(text):
    """
    检测用户输入中的目标语言，支持上下文记忆
    """
    keywords = {
        "英语": ["英语", "英文", "english"],
        "日语": ["日语", "日文", "japanese"],
        "韩语": ["韩语", "韩文", "korean"],
        "法语": ["法语", "法文", "french"],
        "德语": ["德语", "德文", "german"],
        "西班牙语": ["西班牙语", "spanish"],
        "俄语": ["俄语", "俄文", "russian"],
    }
    
    # 上下文相关关键词（表示"使用之前的语言"）
    context_keywords = ["继续", "上面提到的", "之前提到的", "刚才", "同样的", "一样的", "刚才说的", "之前说"]
    
    text_lower = text.lower()
    
    # 1. 首先尝试检测具体语言
    detected = [lang for lang, keys in keywords.items() if any(k in text_lower for k in keys)]
    
    if detected:
        return detected, False  # False 表示不是上下文引用
    
    # 2. 如果没有检测到具体语言，检查是否是上下文引用
    is_context_reference = any(k in text for k in context_keywords)
    
    if is_context_reference:
        # 尝试从 session_state 获取之前的目标语言
        if "last_langs" in st.session_state and st.session_state.last_langs:
            return st.session_state.last_langs, True  # True 表示是上下文引用
    
    return [], False


def find_text_column(df):
    for col in df.columns:
        if any(k in str(col).lower() for k in ["中文", "内容", "文本", "原文", "text"]):
            return col
    return df.columns[0]


def translate_batch(texts, target_lang):
    prompt = f"""将以下文本翻译成{target_lang}，按JSON返回：
{{"translations": ["翻译1", "翻译2", ...]}}

文本：{json.dumps(texts, ensure_ascii=False)}"""
    
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"你是{target_lang}翻译专家"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        content = response.choices[0].message.content
        try:
            result = json.loads(content)
            if "translations" in result:
                return result["translations"][:len(texts)]
        except:
            pass
        
        # 备用解析
        match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if "translations" in result:
                    return result["translations"][:len(texts)]
            except:
                pass
        
        return texts
    except Exception as e:
        st.error(f"翻译出错：{e}")
        return texts


def process_translation(df, text_col, langs, api_key):
    openai.api_key = api_key
    openai.api_base = "https://api.deepseek.com"
    
    result = df.copy()
    texts = df[text_col].astype(str).tolist()
    batch_size = 20
    total = len(texts)
    
    progress_placeholder = st.empty()
    
    for lang in langs:
        with progress_placeholder.container():
            st.write(f"🔄 翻译 {lang}...")
            bar = st.progress(0)
        
        translations = []
        batches = (total + batch_size - 1) // batch_size
        
        for i in range(batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total)
            batch = texts[start:end]
            
            translations.extend(translate_batch(batch, lang))
            bar.progress((i + 1) / batches)
        
        result[f"{lang}_翻译"] = translations
    
    progress_placeholder.empty()
    return result


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="AI 翻译",
    page_icon="🌐",
    layout="centered"
)

# 自定义样式
st.markdown("""
<style>
    .block-container { max-width: 700px; padding: 2rem 1rem; }
    .stChatMessage { padding: 0.3rem 0; }
    
    /* 用户消息靠右 */
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][data-testid="stChatMessageContent-user"]) {
        flex-direction: row-reverse !important;
    }
    .stChatMessage [data-testid="stChatMessageContent"][data-testid="stChatMessageContent-user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 18px 18px 4px 18px;
        margin-left: auto;
        margin-right: 0;
    }
    
    /* 助手消息靠左 */
    .stChatMessage [data-testid="stChatMessageContent"][data-testid="stChatMessageContent-assistant"] {
        background: #f0f2f6;
        border-radius: 18px 18px 18px 4px;
    }
    
    div[data-testid="stFileUploader"] { border: 2px dashed #ddd; border-radius: 8px; padding: 1rem; }
    .stDownloadButton button { width: 100%; background: #4CAF50; color: white; border: none; padding: 0.8rem; border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("## 🔐 API 配置")
    api_key = st.text_input("DeepSeek API Key", type="password", placeholder="sk-...")
    
    st.markdown("---")
    st.markdown("## 📤 上传文件")
    uploaded_file = st.file_uploader("CSV 文件", type=["csv"])
    
    if uploaded_file:
        try:
            df_preview = read_csv_with_encoding(uploaded_file)
            st.success(f"✓ {len(df_preview)} 行数据")
        except Exception as e:
            st.error(f"读取失败：{e}")


# ==================== 主界面 ====================
st.markdown("<h1 style='text-align:center;'>🌐 AI 翻译 Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>上传 CSV，输入目标语言，一键批量翻译</p>", unsafe_allow_html=True)


# 初始化
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！上传 CSV 后告诉我目标语言，例如：*翻译成英语和日语*"}
    ]
if "last_langs" not in st.session_state:
    st.session_state.last_langs = []
if "translation_history" not in st.session_state:
    st.session_state.translation_history = []


# 显示消息 - 用户靠右，助手靠左
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# 聊天输入
if user_input := st.chat_input("输入翻译需求..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        if not api_key:
            st.warning("⚠️ 请输入 API Key")
        elif not uploaded_file:
            st.warning("⚠️ 请上传 CSV 文件")
        else:
            langs, is_context = detect_languages(user_input)
            
            if not langs:
                st.info("🤔 我只能帮你进行翻译哦，快告诉我你需要翻译成什么语言吧")
            else:
                # 如果是上下文引用，显示提示
                if is_context:
                    st.info(f"💡 使用你之前提到的语言：{', '.join(langs)}")
                else:
                    st.success(f"🎯 {', '.join(langs)}")
                
                # 保存当前语言到 session_state（记忆功能）
                st.session_state.last_langs = langs
                
                try:
                    df = read_csv_with_encoding(uploaded_file)
                    col = find_text_column(df)
                    
                    st.caption(f"📄 {col} | {len(df)} 行")
                    
                    with st.expander("预览"):
                        st.dataframe(df[[col]].head(3), use_container_width=True)
                    
                    with st.spinner("翻译中..."):
                        result = process_translation(df, col, langs, api_key)
                    
                    # 保存到历史记录
                    history_item = {
                        "langs": langs,
                        "result": result,
                        "text_col": col,
                        "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
                    }
                    st.session_state.translation_history.insert(0, history_item)
                    
                    st.success(f"✅ 完成！{len(df)} 条 → {len(langs)} 种语言")
                    
                    with st.expander("结果预览"):
                        st.dataframe(result.head(5), use_container_width=True)
                    
                    csv = result.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="⬇️ 下载翻译结果",
                        data=csv,
                        file_name="translated.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"错误：{e}")

# ==================== 历史记录区域 ====================
if st.session_state.translation_history:
    st.markdown("---")
    st.markdown("### 📜 翻译历史")
    
    for idx, item in enumerate(st.session_state.translation_history):
        with st.expander(f"🕐 {item['timestamp']} - {', '.join(item['langs'])} ({len(item['result'])} 条)", expanded=(idx == 0)):
            st.dataframe(item['result'].head(5), use_container_width=True)
            
            csv = item['result'].to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label=f"⬇️ 下载结果 #{idx+1}",
                data=csv,
                file_name=f"translated_{idx+1}_{'_'.join(item['langs'])}.csv",
                mime="text/csv",
                key=f"download_{idx}"
            )
