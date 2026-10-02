import streamlit as st

from main import app


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Deep Research AI",
    page_icon="🔎",
    layout="wide",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🔎 Deep Research AI")
st.caption("Multi-Agent Research System powered by LangGraph")


# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

topic = st.text_area(
    "Research Topic",
    placeholder="Enter a topic you want to research...",
    height=120,
)


research_button = st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# Run Research
# ---------------------------------------------------------

if research_button:

    if not topic.strip():
        st.warning("Please enter a research topic.")
        st.stop()

    initial_state = {
        "topic": topic.strip(),
        "messages": [],
        "searcher": "",
        "reader": "",
        "writer": "",
        "critic": "",
        "is_approved": False,
        "attempt": 0,
        "search_attempts": 0,
        "read_attempts": 0,
        "reader_start": 0,
    }

    with st.spinner("Researching..."):

        try:
            result = app.invoke(
                initial_state,
                config={"recursion_limit": 50}
            )

        except Exception as e:
            st.error(f"Research failed: {str(e)}")
            st.stop()

    st.success("Research completed.")


    # -----------------------------------------------------
    # Workflow Status
    # -----------------------------------------------------

    st.subheader("Workflow Status")

    col1, col2 = st.columns(2)

    with col1:
        if result["is_approved"]:
            st.success("Report Approved")
        else:
            st.warning("Report Not Approved")

    with col2:
        st.info(f"Revision Attempts: {result['attempt']}")


    # -----------------------------------------------------
    # Search Results
    # -----------------------------------------------------

    st.subheader("🔎 Search Results")

    with st.expander("View Search Results", expanded=False):
        st.markdown(result["searcher"])


    # -----------------------------------------------------
    # Reader Results
    # -----------------------------------------------------

    st.subheader("📚 Research / Reader Results")

    with st.expander("View Detailed Research", expanded=False):
        st.markdown(result["reader"])


    # -----------------------------------------------------
    # Final Report
    # -----------------------------------------------------

    st.subheader("📝 Final Report")

    st.markdown(result["writer"])


    # -----------------------------------------------------
    # Critic Review
    # -----------------------------------------------------

    st.subheader("🧐 Critic Review")

    with st.expander("View Critic Review", expanded=False):
        st.markdown(result["critic"])