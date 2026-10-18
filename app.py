
import os
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# 1. STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="News & Weather AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Add these styles inside your existing <style> block

        /* Fix chat input text visibility */
        [data-testid="stChatInput"] textarea {
            color: #111827 !important;
            background-color: #ffffff !important;
            caret-color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #6b7280 !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] {
            background-color: #ffffff !important;
            border: 1px solid #64748b !important;
            border-radius: 12px !important;
        }

        [data-testid="stChatInput"] textarea:focus {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            background-color: #ffffff !important;
        }


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(
                135deg, #0b1220 0%, #111827 55%, #101827 100%
            );
            color: #f8fafc;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: #0b1120;
            border-right: 1px solid #253247;
        }

        .hero {
            padding: 28px 30px;
            border-radius: 22px;
            background: linear-gradient(120deg, #172554, #123047);
            border: 1px solid #294567;
            margin-bottom: 22px;
        }

        .hero h1 {
            color: #f8fafc;
            font-size: 34px;
            margin-bottom: 8px;
        }

        .hero p {
            color: #cbd5e1;
            font-size: 16px;
        }

        .feature-card {
            background: #151f31;
            border: 1px solid #2b3a50;
            border-radius: 15px;
            padding: 18px;
            min-height: 130px;
        }

        .feature-card h3 {
            margin-top: 0;
            color: #f8fafc;
            font-size: 18px;
        }

        .feature-card p {
            color: #b9c6d8;
            font-size: 14px;
        }

        .stChatMessage {
            border: 1px solid #253247;
            border-radius: 14px;
            padding: 14px;
            background: rgba(21, 31, 49, 0.8);
        }

        [data-testid="stChatInput"] {
            border-color: #40516b;
        }

        .small-muted {
            color: #94a3b8;
            font-size: 13px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. READ STREAMLIT SECRETS
# =========================================================

def get_secret(name: str, default: str = "") -> str:
    """Read a key from Streamlit Secrets, with env fallback."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default

    return str(value or os.getenv(name, default)).strip()


GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
NEWS_API_KEY = get_secret("NEWS_API_KEY")
OPENWEATHER_API_KEY = get_secret("OPENWEATHER_API_KEY")

GEMINI_MODEL = get_secret(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)


# =========================================================
# 4. CUSTOM TOOL: LATEST NEWS
# =========================================================

@tool
def get_latest_news(topic: str) -> str:
    """
    Fetch recent English-language news articles for a topic.
    Returns headlines, publisher, publication time, summary and URL.
    Use this for requests about latest or recent news.
    """
    if not topic.strip():
        return "Please provide a topic for the news search."

    url = "https://newsapi.org/v2/everything"

    from_date = (
        datetime.now(timezone.utc) - timedelta(days=3)
    ).date().isoformat()

    params = {
        "q": topic,
        "from": from_date,
        "sortBy": "publishedAt",
        "pageSize": 5,
        "language": "en",
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers={"X-Api-Key": NEWS_API_KEY},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            return "News API error: " + str(
                data.get("message", "Unknown error")
            )

        articles = data.get("articles", [])

        if not articles:
            return f"No recent articles found for {topic!r}."

        output = []

        for i, article in enumerate(articles, start=1):
            source = article.get("source") or {}

            output.append(
                f"{i}. {article.get('title') or 'Untitled'}\n"
                f"Publisher: {source.get('name', 'Unknown')}\n"
                f"Published at (UTC): "
                f"{article.get('publishedAt', 'Unknown')}\n"
                f"Summary: "
                f"{article.get('description') or 'No summary available'}\n"
                f"URL: {article.get('url') or 'Unavailable'}"
            )

        return "\n\n".join(output)

    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 401:
            return "News API authentication failed. Check NEWS_API_KEY."
        if status == 429:
            return "News API rate limit reached. Please try again later."
        return f"News API HTTP error: {exc}"

    except requests.RequestException as exc:
        return f"Could not fetch news due to a network error: {exc}"


# =========================================================
# 5. CUSTOM TOOL: CURRENT WEATHER
# =========================================================

@tool
def get_weather(city: str) -> str:
    """
    Fetch current weather for a city.
    Returns weather description, temperature in Celsius,
    feels-like temperature, humidity and wind speed.
    """
    if not city.strip():
        return "Please provide a city name."

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        main = data["main"]
        weather = data["weather"][0]
        wind = data.get("wind", {})
        country = data.get("sys", {}).get("country", "")

        return (
            f"City: {data.get('name', city)}, {country}\n"
            f"Conditions: {weather.get('description', 'Unknown')}\n"
            f"Temperature: {main.get('temp')} °C\n"
            f"Feels like: {main.get('feels_like')} °C\n"
            f"Minimum: {main.get('temp_min')} °C\n"
            f"Maximum: {main.get('temp_max')} °C\n"
            f"Humidity: {main.get('humidity')}%\n"
            f"Wind speed: {wind.get('speed', 'Unknown')} m/s"
        )

    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None

        if status == 401:
            return "Weather API authentication failed. Check OPENWEATHER_API_KEY."
        if status == 404:
            return f"City {city!r} was not found. Check the spelling."
        if status == 429:
            return "Weather API rate limit reached. Try again later."

        return f"Weather API HTTP error: {exc}"

    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        return f"Could not fetch weather data: {exc}"


# =========================================================
# 6. CREATE THE LANGCHAIN AGENT
# =========================================================

@st.cache_resource(show_spinner=False)
def build_agent(api_key: str, model_name: str):
    """Create and cache the model and agent for this app process."""

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
    )

    return create_agent(
        model=llm,
        tools=[
            get_latest_news,
            get_weather,
        ],
        system_prompt="""
You are News & Weather AI, a helpful assistant.

Available tools:
- get_latest_news(topic): fetch recent news articles.
- get_weather(city): fetch current weather conditions.

Instructions:
1. Call the news tool when current or recent news is requested.
2. Call the weather tool when current weather is requested.
3. If both are requested, use both tools.
4. Never invent article titles, sources, dates, URLs or weather readings.
5. Clearly explain when the tools return no data or an error.
6. Present news as readable bullet points and include available links.
7. Present weather in a concise, easy-to-read format.
8. Ask a short clarification when a required topic or city is missing.
9. Do not claim to have searched live data unless the relevant tool ran.
""",
    )


# =========================================================
# 7. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🌍 News & Weather AI")
    st.caption("Your personal real-time information assistant")

    st.divider()

    st.markdown("### 🧰 Available tools")
    st.markdown("- 📰 Latest News")
    st.markdown("- 🌤️ Current Weather")

    st.divider()

    st.markdown("### 💡 Try an example")

    examples = [
        "Latest AI news",
        "Weather in Delhi",
        "Latest Java news and weather in Bengaluru",
        "Latest business news",
    ]

    for example in examples:
        if st.button(
            example,
            key=f"example_{example}",
            use_container_width=True,
        ):
            st.session_state.pending_prompt = example

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_prompt = ""
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<p class="small-muted">'
        'Powered by LangChain, Gemini, NewsAPI and OpenWeatherMap.'
        '</p>',
        unsafe_allow_html=True,
    )


# =========================================================
# 8. MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌍 News & Weather AI Agent</h1>
        <p>
            One intelligent assistant for the latest headlines
            and current weather conditions around the world.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

feature1, feature2, feature3 = st.columns(3)

with feature1:
    st.markdown(
        """
        <div class="feature-card">
            <h3>📰 Latest News</h3>
            <p>Search recent articles by topic and open the original source.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feature2:
    st.markdown(
        """
        <div class="feature-card">
            <h3>🌤️ Live Weather</h3>
            <p>Get current temperature, conditions, humidity and wind speed.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feature3:
    st.markdown(
        """
        <div class="feature-card">
            <h3>🤖 AI Agent</h3>
            <p>Ask naturally. The agent decides which custom tools to use.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### 💬 Chat with your agent")


# =========================================================
# 9. CHAT STATE AND HISTORY
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🌍"):
        st.markdown(
            """
            Hello! 👋 I'm your News & Weather AI agent.

            You can ask me questions like:

            - 📰 What are the latest AI developments?
            - 🌤️ What's the weather in Mumbai?
            - 🌍 Give me the latest technology news and weather in Delhi.
            """
        )

# Render the previous conversation.
for message in st.session_state.messages:
    with st.chat_message(
        message["role"],
        avatar="🧑" if message["role"] == "user" else "🌍",
    ):
        st.markdown(message["content"])


# =========================================================
# 10. PROCESS USER INPUT
# =========================================================

typed_prompt = st.chat_input(
    "Ask about news, weather, or both..."
)

pending_prompt = st.session_state.pending_prompt
st.session_state.pending_prompt = ""

user_prompt = typed_prompt or pending_prompt

if user_prompt:
    # Display and store the user message.
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_prompt)

    # Check configuration before building the agent.
    missing_keys = []

    if not GOOGLE_API_KEY:
        missing_keys.append("GOOGLE_API_KEY")
    if not NEWS_API_KEY:
        missing_keys.append("NEWS_API_KEY")
    if not OPENWEATHER_API_KEY:
        missing_keys.append("OPENWEATHER_API_KEY")

    if missing_keys:
        answer = (
            "### Configuration required\n\n"
            "Please add the following keys in Streamlit Secrets:\n\n"
            + "\n".join(f"- `{key}`" for key in missing_keys)
        )

        with st.chat_message("assistant", avatar="🌍"):
            st.error(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    else:
        with st.chat_message("assistant", avatar="🌍"):
            try:
                with st.spinner(
                    "🤖 Analyzing your question and fetching data..."
                ):
                    agent = build_agent(
                        GOOGLE_API_KEY,
                        GEMINI_MODEL,
                    )

                    # Include conversation history for follow-up questions.
                    conversation = [
                        {
                            "role": msg["role"],
                            "content": msg["content"],
                        }
                        for msg in st.session_state.messages
                    ]

                    result = agent.invoke(
                        {"messages": conversation}
                    )

                    final_message = result["messages"][-1]
                    content = final_message.content

                    if isinstance(content, str):
                        answer = content
                    else:
                        # Gemini may return content blocks.
                        answer = "\n\n".join(
                            str(block.get("text", ""))
                            for block in content
                            if isinstance(block, dict)
                            and block.get("text")
                        )

                    if not answer.strip():
                        answer = "The agent returned an empty response. Please try again."

                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as exc:
                st.error(
                    "The agent encountered an error. Check your "
                    "Gemini configuration, API limits, and application logs."
                )
                st.caption(f"Error details: {exc}")
