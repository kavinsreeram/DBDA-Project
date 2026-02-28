import streamlit as st
from full_fact_check_pipeline import fact_check

st.set_page_config(
    page_title="RAG News Fact Retriever",
    page_icon="📰",
    layout="centered"
)

# Custom CSS Styling
st.markdown("""
    <style>
    .title-style {
        font-size:24px;
        font-weight:bold;
        color:#1f77b4;
    }
    .section-header {
        font-size:20px;
        font-weight:bold;
        margin-top:20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-style'>📰 RAG-Based News Fact Retriever</div>", unsafe_allow_html=True)

claim = st.text_input(
    "Enter a news claim or topic",
    placeholder="Example: Government announced free petrol for all vehicles"
)

if st.button("Search & Summarize"):

    if not claim.strip():
        st.warning("⚠️ Please enter a claim.")
    else:
        with st.spinner("🔍 Processing..."):
            result = fact_check(claim)

        # Card-style display
        st.markdown("---")

        st.markdown("<div class='section-header'>📰 Title</div>", unsafe_allow_html=True)
        st.info(result["title"])

        st.markdown("<div class='section-header'>📝 Claim</div>", unsafe_allow_html=True)
        st.write(result["claim"])

        st.markdown("<div class='section-header'>📌 Summary</div>", unsafe_allow_html=True)
        st.success(result["summary"])

        st.markdown("<div class='section-header'>🔗 Top Sources</div>", unsafe_allow_html=True)
        for i, src in enumerate(result["sources"], 1):
            st.markdown(f"{i}. [{src}]({src})")

st.markdown("---")
st.caption("Built using RAG pipeline with trusted news sources and LLM summarization")
