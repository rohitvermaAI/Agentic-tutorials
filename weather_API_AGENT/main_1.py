import requests
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(".env")

# OpenWeatherMap API key
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_weather(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"  # Celsius
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if response.status_code != 200:
        return None, f"Sorry, I couldn't fetch weather for '{city}'. Reason: {data.get('message', 'Unknown error')}"

    weather_info = {
        "description": data["weather"][0]["description"].capitalize(),
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"]
    }  ## Dictionary of the data returned by WeatherAPI

    report = (
        f"The weather in {city} is currently {weather_info['description']}.\n"
        f"Temperature: {weather_info['temp']}°C (feels like {weather_info['feels_like']}°C)\n"
        f"Humidity: {weather_info['humidity']}%"
    )
    
    return weather_info, report

def get_ai_suggestions(weather_info, city):
    with open("prompt.txt") as f:
        prompt = f.read()
    print("prompt before formatting: ", prompt)
    prompt = prompt.format(
        city=city,
        description=weather_info["description"],
        temp=weather_info["temp"],
        feels_like=weather_info["feels_like"],
        humidity=weather_info["humidity"]
    )
    print(f"prompt after formatting: {prompt}")
    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )
    
    return response.output_text

if __name__ == "__main__":
    while True:
        city = input("Enter a city name (or type 'exit' to quit): ")
        if city.lower() == "exit":
            print("Goodbye! ☀️")
            break
        
        weather_info, report = get_weather(city)
        if weather_info:
            print("\n📍 Weather Report:")
            print(report)
            print("\n💡 AI Suggestions:")
            print(get_ai_suggestions(weather_info, city))
        else:
            print(report)
