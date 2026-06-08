from strands import Agent, tool
from strands.models.openai import OpenAIModel
from datetime import datetime
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()
OPEN_AI_URL = os.getenv("OPEN_AI_URL")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

@tool
def get_weather(city: str) -> dict:
    weather_data = {
        "San Francisco": {"temp_f": 62, "conditions": "Partly Cloudy", "humidity": 75},
        "New York": {"temp_f": 78, "conditions": "Sunny", "humidity": 55},
        "Seattle": {"temp_f": 58, "conditions": "Rainy", "humidity": 85},
        "Austin": {"temp_f": 95, "conditions": "Hot and Sunny", "humidity": 40},
    }

    result = weather_data.get(city, {"temp_f": 70, "conditions": "Unknown", "humidity": 50})
    result["city"] = city
    result["timestamp"] = datetime.now().isoformat()
    return result


@tool
def calculate_travel_time(origin: str, destination: str, mode: str = "driving") -> dict:
    base_times = {
        ("San Francisco", "San Jose"): {"driving": 45, "transit": 75, "walking": 600},
        ("San Francisco", "Oakland"): {"driving": 25, "transit": 40, "walking": 180},
        ("New York", "Brooklyn"): {"driving": 30, "transit": 25, "walking": 120},
        ("Ferry Building", "SFJAZZ Center"): {"driving": 12, "transit": 20, "walking": 35},
    }

    key = (origin, destination)
    reverse_key = (destination, origin)

    if key in base_times:
        times = base_times[key]
    elif reverse_key in base_times:
        times = base_times[reverse_key]
    else:
        times = {"driving": 30, "transit": 45, "walking": 90}

    duration_minutes = times.get(mode, times["driving"])

    return {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "duration_minutes": duration_minutes,
        "duration_text": (
            f"{duration_minutes} minutes"
            if duration_minutes < 60
            else f"{duration_minutes // 60}h {duration_minutes % 60}m"
        ),
    }


@tool
def get_local_events(city: str, category: Optional[str] = None) -> list:
    all_events = {
        "San Francisco": [
            {"name": "Jazz Night at SFJAZZ", "category": "music", "time": "8:00 PM", "venue": "SFJAZZ Center"},
            {"name": "Giants vs Dodgers", "category": "sports", "time": "7:15 PM", "venue": "Oracle Park"},
            {"name": "Street Food Festival", "category": "food", "time": "11:00 AM", "venue": "Ferry Building"},
            {"name": "AI Meetup", "category": "tech", "time": "6:30 PM", "venue": "Moscone Center"},
        ],
        "New York": [
            {"name": "Broadway Show: Hamilton", "category": "music", "time": "7:30 PM", "venue": "Richard Rodgers Theatre"},
            {"name": "Yankees Game", "category": "sports", "time": "7:05 PM", "venue": "Yankee Stadium"},
            {"name": "Smorgasburg Food Market", "category": "food", "time": "11:00 AM", "venue": "Williamsburg"},
        ],
    }

    events = all_events.get(city, [])

    if category:
        events = [e for e in events if e["category"] == category.lower()]

    return events


def create_agent() -> Agent:
    model = OpenAIModel(
        client_args={
            "api_key": OPEN_AI_API_KEY,
            "base_url": OPEN_AI_URL,
            "default_headers": {
                "HTTP-Referer": "http://localhost",
                "X-Title": "Strands OpenRouter Agent",
            },
        },
        model_id="openai/gpt-4o-mini",
        params={
            "temperature": 0.3,
        },
    )

    system_prompt = """
You are a helpful local guide assistant. You help users plan their day by providing
weather information, travel times, and local events.

When a user asks about planning activities:
1. First check the weather.
2. Look up relevant local events.
3. Calculate travel times if needed.
4. Synthesize all information into a practical recommendation.

Be concise and practical.
"""

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_weather, calculate_travel_time, get_local_events],
    )


def main():
    agent = create_agent()

    user_query = """
I'm in San Francisco and want to do something fun this evening.
What's the weather like, and are there any music events happening?
If there's something good, how long would it take me to get there from the Ferry Building?
"""

    response = agent(user_query)
    print(response)


if __name__ == "__main__":
    main()