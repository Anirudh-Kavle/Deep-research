"""Workflow module for advanced-agent research pipeline."""

import os
import json
from typing import Dict, Any
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from src.models import ResearchState, CompanyInfo, CompanyAnalysis
from src.firecrawl_service import FirecrawlService
from src.prompts import DeveloperToolsPrompts

GROQ_MODEL = os.getenv(
    "GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)

groq_model = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


class Workflow:
    def __init__(self):
        self.firecrawl = FirecrawlService()
        self.llm = groq_model
        self.prompts = DeveloperToolsPrompts()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(ResearchState)
        graph.add_node("extract_tools", self._extract_tools_step)
        graph.add_node("research", self._research_step)
        graph.add_node("analyze", self._analyze_step)
        graph.set_entry_point("extract_tools")
        graph.add_edge("extract_tools", "research")
        graph.add_edge("research", "analyze")
        graph.add_edge("analyze", END)
        return graph.compile()

    def _extract_tools_step(self, state: ResearchState) -> Dict[str, Any]:
        print(f"🔍 Finding articles about: {state.query}")

        article_query = f"{state.query} tools comparison best alternatives"
        search_results = self.firecrawl.search_companies(
            article_query, num_results=3)

        all_content = ""
        # Safely handle search_results which may be a list (error case) or object with .data
        results_data = search_results.data if hasattr(search_results, 'data') else []
        for result in results_data:
            url = result.get("url", "")
            if url.endswith(".pdf"):
                continue
            scraped = self.firecrawl.scrape_company_pages(url)
            if scraped and hasattr(scraped, "markdown") and isinstance(scraped.markdown, str) and scraped.markdown.strip():
                all_content += scraped.markdown[:1500] + "\n\n"

        messages = [
            SystemMessage(content=self.prompts.TOOL_EXTRACTION_SYSTEM),
            HumanMessage(content=self.prompts.tool_extraction_user(
                state.query, all_content))
        ]
        try:
            response = self.llm.invoke(messages)
            tool_names = []
            if isinstance(response.content, str):
                for line in response.content.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # Filter out lines that are not likely tool names
                    if line.lower().startswith("here are"):
                        continue
                    if line[0].isdigit() and "." in line:
                        continue
                    summary_keywords = ["top", "mentioned", "alternatives", "tools", "summary", "article", "content"]
                    if any(word in line.lower() for word in summary_keywords):
                        continue
                    if len(line.split()) > 5:
                        continue
                    tool_names.append(line)
            print(f"Extracted tools: {', '.join(tool_names[:5])}")
            return {"extracted_tools": tool_names}
        except Exception as e:
            print(e)
            return {"extracted_tools": []}

    def _analyze_company_content(self, company_name: str, content: str) -> CompanyAnalysis:
        # Try analysis with retries
        max_retries = 2
        for attempt in range(max_retries):
            try:
                messages = [
                    SystemMessage(content=self.prompts.TOOL_ANALYSIS_SYSTEM),
                    HumanMessage(content=self.prompts.tool_analysis_user(
                        company_name, content))
                ]

                response = self.llm.invoke(messages)
                json_str = response.content.strip()

                if attempt == 0:  # Only print debug for first attempt
                    print(
                        f"📝 Raw response for {company_name}: {json_str[:200]}...")

                # Check if response is empty
                if not json_str:
                    if attempt < max_retries - 1:
                        print(
                            f"⚠️ Empty response for {company_name}, retrying...")
                        continue
                    else:
                        raise ValueError("Empty response from LLM")

                # Clean up common JSON formatting issues
                if json_str.startswith('```json'):
                    json_str = json_str[7:]
                if json_str.endswith('```'):
                    json_str = json_str[:-3]
                json_str = json_str.strip()

                # If still not JSON, try to extract JSON from text
                if not json_str.startswith('{'):
                    # Look for JSON within the response
                    start_idx = json_str.find('{')
                    end_idx = json_str.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = json_str[start_idx:end_idx+1]
                    else:
                        if attempt < max_retries - 1:
                            print(
                                f"⚠️ No JSON found for {company_name}, retrying...")
                            continue
                        else:
                            # Fallback: create basic analysis from text
                            return self._create_fallback_analysis(company_name, json_str, content)

                # Parse JSON
                data = json.loads(json_str)

                # Convert string booleans to actual booleans
                if isinstance(data.get('is_open_source'), str):
                    if data['is_open_source'].lower() == 'true':
                        data['is_open_source'] = True
                    elif data['is_open_source'].lower() == 'false':
                        data['is_open_source'] = False
                    else:
                        data['is_open_source'] = None

                if isinstance(data.get('api_available'), str):
                    if data['api_available'].lower() == 'true':
                        data['api_available'] = True
                    elif data['api_available'].lower() == 'false':
                        data['api_available'] = False
                    else:
                        data['api_available'] = None

                # Ensure lists are actually lists
                for field in ['tech_stack', 'language_support', 'integration_capabilities']:
                    if field not in data:
                        data[field] = []
                    elif not isinstance(data[field], list):
                        data[field] = []

                # Ensure required string fields exist
                if 'pricing_model' not in data:
                    data['pricing_model'] = "Unknown"
                if 'description' not in data:
                    data['description'] = "No description available"

                print(f"✅ Successfully analyzed {company_name}")
                return CompanyAnalysis(**data)

            except Exception as e:
                if attempt < max_retries - 1:
                    print(
                        f"⚠️ Analysis attempt {attempt + 1} failed for {company_name}: {str(e)}, retrying...")
                    continue
                else:
                    print(
                        f"⚠️ All analysis attempts failed for {company_name}: {str(e)}")
                    return self._create_fallback_analysis(company_name, "", content)

    def _create_fallback_analysis(self, company_name: str, llm_response: str, content: str) -> CompanyAnalysis:
        """Create basic analysis when JSON parsing fails"""
        # Try to extract basic info from content
        description = f"{company_name} is a developer tool"
        pricing_model = "Unknown"

        # Simple keyword detection for pricing
        content_lower = content.lower()
        if any(word in content_lower for word in ["free", "$0", "no cost"]):
            pricing_model = "Free"
        elif any(word in content_lower for word in ["freemium", "free tier", "free plan"]):
            pricing_model = "Freemium"
        elif any(word in content_lower for word in ["enterprise", "contact sales"]):
            pricing_model = "Enterprise"
        elif any(word in content_lower for word in ["$", "price", "paid", "subscription"]):
            pricing_model = "Paid"

        # Extract basic info from company name
        tech_stack = []
        language_support = []
        integration_capabilities = []

        if "github" in company_name.lower():
            integration_capabilities = ["GitHub", "VS Code"]
        if "copilot" in company_name.lower():
            language_support = ["Python", "JavaScript", "TypeScript"]
        if "cursor" in company_name.lower():
            integration_capabilities = ["VS Code"]
        if "tabnine" in company_name.lower():
            integration_capabilities = ["VS Code", "IntelliJ", "Sublime Text"]

        return CompanyAnalysis(
            pricing_model=pricing_model,
            is_open_source=None,
            tech_stack=tech_stack,
            description=description,
            api_available=None,
            language_support=language_support,
            integration_capabilities=integration_capabilities,
        )

    def _research_step(self, state: ResearchState) -> Dict[str, Any]:
        extracted_tools = getattr(state, "extracted_tools", [])

        if not extracted_tools:
            print("⚠️ No extracted tools found, falling back to direct search")
            search_results = self.firecrawl.search_companies(
                state.query, num_results=4)
            results_data = search_results.data if hasattr(search_results, 'data') else []
            tool_names = [
                result.get("metadata", {}).get("title", "Unknown")
                for result in results_data
            ]
        else:
            tool_names = extracted_tools[:4]

        print(f"🔬 Researching specific tools: {', '.join(tool_names)}")

        companies = []
        for tool_name in tool_names:
            tool_search_results = self.firecrawl.search_companies(
                tool_name + " official site", num_results=1)

            tool_results_data = tool_search_results.data if hasattr(tool_search_results, 'data') else []
            if tool_results_data:
                result = tool_results_data[0]
                url = result.get("url", "")

                company = CompanyInfo(
                    name=tool_name,
                    description=result.get("markdown", ""),
                    website=url,
                    tech_stack=[],
                    competitors=[]
                )

                scraped = self.firecrawl.scrape_company_pages(url)
                # Ensure content is a string for _analyze_company_content
                content = scraped.markdown if (scraped and hasattr(scraped, "markdown") and isinstance(scraped.markdown, str)) else ""
                analysis = self._analyze_company_content(
                    company.name, content)

                company.pricing_model = analysis.pricing_model
                company.is_open_source = analysis.is_open_source
                company.tech_stack = analysis.tech_stack
                company.description = analysis.description
                company.api_available = analysis.api_available
                company.language_support = analysis.language_support
                company.integration_capabilities = analysis.integration_capabilities

                companies.append(company)

        return {"companies": companies}

    def _analyze_step(self, state: ResearchState) -> Dict[str, Any]:
        print("Generating recommendations")

        company_data = ", ".join([
            company.json() for company in state.companies
        ])

        messages = [
            SystemMessage(content=self.prompts.RECOMMENDATIONS_SYSTEM),
            HumanMessage(content=self.prompts.recommendations_user(
                state.query, company_data))
        ]

        response = self.llm.invoke(messages)
        return {"analysis": response.content}

    def run(self, query: str) -> ResearchState:
        initial_state = ResearchState(query=query)
        final_state = self.workflow.invoke(initial_state)
        return ResearchState(**final_state)

