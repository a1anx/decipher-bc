import requests

BASE_URL = "https://api.figshare.com/v2"
item_id = 12420968

response = requests.get(
    f"{BASE_URL}/articles/{item_id}/files",
    params={"limit": 100},  # default page size is small, so ask for more
    timeout=30,
)
response.raise_for_status()


from pathlib import Path

target_name = "Immune_ALL_human.h5ad"


if __name__ == "__main__":
    # for file_record in response.json():
    #     size_in_megabytes = file_record["size"] / 1e6
    #     print(file_record["id"], file_record["name"], f"{size_in_megabytes:.0f} MB")
    target_record = next(
        file_record for file_record in response.json()
        if file_record["name"] == target_name
    )

    output_path = Path(target_name)
    with requests.get(target_record["download_url"], stream=True, timeout=60) as download_response:
        download_response.raise_for_status()
        with open(output_path, "wb") as output_file:
            for chunk in download_response.iter_content(chunk_size=1024 * 1024):
                output_file.write(chunk)

    print("Saved", output_path, output_path.stat().st_size / 1e6, "MB")