import json
from constants import RACES_FILE_PATH
from models.stirnu_buks import parse_stirnu_buks_races


if __name__ == "__main__":
    races = json.loads(RACES_FILE_PATH.read_text())

    races_sb_raw = races.get("stirnu_buks")
    races_sb = parse_stirnu_buks_races(races_sb_raw)

    first_sb = races_sb[0]
    print(first_sb.location)
    print(first_sb.results[0].model_dump_json())
