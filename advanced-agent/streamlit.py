# main.py
import streamlit as st
import os
from src.workflow import Workflow
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()


def main():
    st.set_page_config(
        page_title="Developer Tools Research Agent", layout="wide")
    st.title("👨‍💻 Developer Tools Research Agent")

    # Sidebar for API key configuration
    st.sidebar.header("API Key Configuration")
    firecrawl_api_key = st.sidebar.text_input(
        "Firecrawl API Key", type="password", key="firecrawl_api_key_input",
        # Pre-fill from .env if available
        value=os.getenv("FIRECRAWL_API_KEY", "")
    )
    groq_api_key = st.sidebar.text_input(
        "Groq API Key", type="password", key="groq_api_key_input",
        value=os.getenv("GROQ_API_KEY", "")  # Pre-fill from .env if available
    )

    # Store API keys in session state
    if firecrawl_api_key:
        st.session_state["FIRECRAWL_API_KEY"] = firecrawl_api_key
    if groq_api_key:
        st.session_state["GROQ_API_KEY"] = groq_api_key

    # Check if API keys are provided
    if not st.session_state.get("FIRECRAWL_API_KEY") or not st.session_state.get("GROQ_API_KEY"):
        st.warning(
            "Please enter your Firecrawl and Groq API keys in the sidebar to proceed.")
        st.stop()  # Stop the app execution until keys are provided

    # Override os.getenv for FirecrawlService and ChatGroq with session state values
    # This is a bit of a workaround for how your current code uses os.getenv.
    # In a real app, you might pass these directly to the Workflow constructor.
    os.environ["FIRECRAWL_API_KEY"] = st.session_state["FIRECRAWL_API_KEY"]
    os.environ["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"]

    # Initialize the workflow
    try:
        workflow = Workflow()
    except ValueError as e:
        st.error(
            f"Error initializing workflow: {e}. Please check your API keys.")
        st.stop()

    st.write(
        "Enter a query to research developer tools (e.g., 'CI/CD tools', 'backend frameworks').")

    query = st.text_input("🔍 Developer Tools Query:",
                          placeholder="e.g., 'code refactoring tools'", key="query_input")

    if st.button("Start Research 🚀"):
        if query:
            with st.spinner("Researching and analyzing developer tools..."):
                try:
                    result = workflow.run(query)
                    st.success("Research complete!")

                    st.subheader(f"📊 Results for: {query}")
                    st.markdown("---")

                    if result.companies:
                        for i, company in enumerate(result.companies, 1):
                            st.markdown(f"### {i}. 🏢 {company.name}")
                            st.write(f"🌐 **Website**: {company.website}")
                            st.write(f"💰 **Pricing**: {company.pricing_model}")
                            st.write(
                                f"📖 **Open Source**: {'✅ Yes' if company.is_open_source else '❌ No' if company.is_open_source is False else 'Unknown'}")

                            if company.tech_stack:
                                st.write(
                                    f"🛠️ **Tech Stack**: {', '.join(company.tech_stack)}")

                            if company.language_support:
                                st.write(
                                    f"💻 **Language Support**: {', '.join(company.language_support)}")

                            if company.api_available is not None:
                                api_status = "✅ Available" if company.api_available else "❌ Not Available"
                                st.write(f"🔌 **API**: {api_status}")

                            if company.integration_capabilities:
                                st.write(
                                    f"🔗 **Integrations**: {', '.join(company.integration_capabilities)}")

                            if company.description and company.description != "Analysis failed":
                                st.info(
                                    f"📝 **Description**: {company.description}")
                            st.markdown("---")
                    else:
                        st.warning(
                            "No relevant companies found for your query.")

                    if result.analysis:
                        st.header("Developer Recommendations:")
                        st.markdown(result.analysis)
                    else:
                        st.warning("Could not generate recommendations.")

                except Exception as e:
                    st.error(f"An error occurred during research: {e}")
        else:
            st.warning("Please enter a query to start the research.")


if __name__ == "__main__":
    main()
