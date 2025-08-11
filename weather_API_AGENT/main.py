import requests

API_KEY = "817503c3b9afa2609e36a6b8239520bd"  # Replace with your API key
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_weather(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"  # Celsius
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if response.status_code != 200:
        return f"Sorry, I couldn't fetch weather for '{city}'. Reason: {data.get('message', 'Unknown error')}"

    weather_desc = data["weather"][0]["description"].capitalize()
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]

    return (
        f"The weather in {city} is currently {weather_desc}.\n"
        f"Temperature: {temp}°C (feels like {feels_like}°C)\n"
        f"Humidity: {humidity}%"
    )

if __name__ == "__main__":
    while True:
        city = input("Enter a city name (or type 'exit' to quit): ")
        if city.lower() == "exit":
            print("Goodbye! ☀️")
            break
        print(get_weather(city))
