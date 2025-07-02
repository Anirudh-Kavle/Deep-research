class DeveloperToolsPrompts:
    """Collection of prompts for analyzing developer tools and technologies"""
    
    # Tool extraction prompts
    TOOL_EXTRACTION_SYSTEM = """You are a tech researcher. Extract specific tool, library, platform, or service names from articles.
                            Focus on actual products/tools that developers can use, not general concepts or features."""
    
    @staticmethod
    def tool_extraction_user(query: str, content: str) -> str:
        return f"""Query: {query}
                Article Content: {content}
                
                Extract a list of specific tool/service names mentioned in this content that are relevant to "{query}".
                
                Rules:
                - Only include actual product names, not generic terms
                - Focus on tools developers can directly use/implement
                - Include both open source and commercial options
                - Limit to the 5 most relevant tools
                - Return just the tool names, one per line, no descriptions
                
                Example format:
                Supabase
                PlanetScale
                Railway
                Appwrite
                Nhost"""
    
    # Company/Tool analysis prompts
    TOOL_ANALYSIS_SYSTEM = """You are analyzing developer tools and programming technologies.
                            Focus on extracting information relevant to programmers and software developers.
                            Pay special attention to programming languages, frameworks, APIs, SDKs, and development workflows.
                            
                            CRITICAL: You must respond with ONLY valid JSON. No explanations, no markdown formatting, no additional text.
                            Start your response immediately with { and end with }."""
    
    @staticmethod
    def tool_analysis_user(company_name: str, content: str) -> str:
        return f"""Analyze: {company_name}
Content: {content[:2000]}

Return ONLY this JSON structure (no other text):
{{
  "pricing_model": "Free",
  "is_open_source": false,
  "tech_stack": ["AI", "Machine Learning"],
  "description": "{company_name} is an AI-powered coding assistant that helps developers write code faster",
  "api_available": true,
  "language_support": ["Python", "JavaScript"],
  "integration_capabilities": ["VS Code", "GitHub"]
}}

Rules:
- pricing_model: "Free", "Freemium", "Paid", "Enterprise", or "Unknown"
- is_open_source: true, false, or null (no quotes)
- Arrays must contain quoted strings
- description: single sentence under 100 characters
- Start response with {{ and end with }}
- No markdown, no explanations, ONLY JSON"""
    
    # Recommendation prompts
    RECOMMENDATIONS_SYSTEM = """You are a senior software engineer providing quick, concise tech recommendations.
                            Keep responses brief and actionable - maximum 3-4 sentences total."""
    
    @staticmethod
    def recommendations_user(query: str, company_data: str) -> str:
        return f"""Developer Query: {query}
                Tools/Technologies Analyzed: {company_data}
                
                Provide a brief recommendation (3-4 sentences max) covering:
                - Which tool is best and why
                - Key cost/pricing consideration
                - Main technical advantage
                
                Be concise and direct - no long explanations needed."""