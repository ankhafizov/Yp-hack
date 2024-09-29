import requests

# пример запроса на проверку дубликата
api_url = "http://62.68.147.104:8000/docs"
check_url = f"{api_url}/check-video-duplicate"
video_path = "https://s3.ritm.media/yappy-db-duplicates/3c9011da-026a-43d4-84c6-1fe7838a14c4.mp4"

check_response = requests.post(check_url, json={"link": video_path})
print(check_response.json())
