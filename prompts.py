from langchain_core.prompts import ChatPromptTemplate

# Searcher
SEARCH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a research assistant.

        Search the web for recent, reliable and relevant information.

        Use the web_search tool to gather information about the topic.
        Perform multiple searches only if needed.

        IMPORTANT: once you have enough relevant results to answer the topic, stop calling tools.
        Do not keep searching in a loop after the topic is already covered.
        Provide a concise synthesis of the research you found."""
    ),
    (
        "human",
        "Research the following topic:\n\n{topic}"
    )
])

# Reaader
READER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a research reader.

        Your job is to examine search results and gather detailed information
        from the most relevant sources.

        Use the scrape_url tool to read the most relevant sources.

        IMPORTANT:
        - Scrape at most 3 URLs.
        - Prefer authoritative and reliable sources.
        - Do not scrape the same URL more than once.
        - Once you have enough information to answer the topic, stop using tools immediately.
        - Do not keep scraping the same site or making repeated tool calls in a loop.
        """
            ),
            (
                "human",
                """Research topic:
        {topic}

        Search results:
        {search_results}

        Read the most relevant sources and gather detailed information."""
    ),
])
# Writer
WRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

    Topic: {topic}

    Research Gathered:
    {research}

    Structure the report as:
    - Introduction
    - Key Findings (minimum 3 well-explained points)
    - Conclusion
    - Sources (list all URLs found in the research)

    Be detailed, factual and professional."""),
])

# Critique
CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

    Report:
    {report}

    Respond in this exact format:

    Score: X/10

    Strengths:
    - ...
    - ...

    Areas to Improve:
    - ...
    - ...

    One line verdict:
    ..."""),
])