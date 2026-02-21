import re
import json
import argparse

from pathlib import Path
from datetime import datetime, time
from typing import Optional, List, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import xml.etree.ElementTree as ET
import trafilatura as tt


class StirnuBuksResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    rank: int = Field(alias="#")
    participant: str = Field(alias="Dalībnieks")
    result: time = Field(alias="Rezultāts")
    age_group: str = Field(alias="Vecuma grupa")
    age_group_rank: Optional[int] = Field(alias="Vieta vecuma grupā", default=None)
    gender_rank: int = Field(alias="Vieta pēc dzimuma")
    behind_winner: time = Field(alias="Pret uzvarētāju")
    behind_age_group_winner: time = Field(alias="Pret vecuma grupas uzvarētāju")
    points: int = Field(alias="Punkti")
    sprint_time: Optional[time] = Field(alias="Sprinta posma laiks", default=None)
    sprint_points: Optional[int] = Field(alias="Sprinta posma punkti", default=None)
    sprint_distance: str = Field(alias="Sprinta distance", exclude=True)
    total_points: int = Field(alias="Kopā")

    @field_validator("rank", mode="before")
    @classmethod
    def parse_rank(cls, v):
        if isinstance(v, str):
            v = v.strip().replace(".", "")
            return int(v)
        elif isinstance(v, int):
            return v
        raise ValueError(f"Failed to parse rank '{v}'")

    @field_validator("participant", mode="after")
    @classmethod
    def parse_name(cls, v: str):
        return v.split("(")[0].strip()

    @field_validator("result", mode="before")
    @classmethod
    def parse_result(cls, v):
        if isinstance(v, time):
            return v
        return datetime.strptime(v, "%H:%M:%S").time()

    @field_validator("gender_rank", mode="before")
    @classmethod
    def parse_gender_rank(cls, v):
        # Extract numeric rank from gender_rank e.g. "V1" -> 1
        match = re.search(r"(\d+)", v)
        if match:
            return int(match.group(0))
        raise ValueError(f"Failed to parse gender rank '{v}'")

    @field_validator("behind_winner", "behind_age_group_winner", mode="before")
    @classmethod
    def parse_gap(cls, v):
        if isinstance(v, time):
            return v
        parts = v.strip().split(":")
        if len(parts) == 2:
            return time(minute=int(parts[0]), second=int(parts[1]))
        elif len(parts) == 3:
            return time(hour=int(parts[0]), minute=int(parts[1]), second=int(parts[2]))
        raise ValueError(f"Failed to parse gap '{v}'")

    @field_validator("points", "total_points", mode="before")
    @classmethod
    def parse_int(cls, v):
        if isinstance(v, str):
            v = v.strip().replace(" ", "")
            return int(v)
        raise ValueError(f"Failed to parse points '{v}'")

    @model_validator(mode="after")
    def extract_composite_fields(self):
        # Extract age_group_rank from e.g. "VL1 (1)"
        match = re.search(r"\((\d+)\)", self.age_group)
        if match:
            self.age_group_rank = int(match.group(1))
            self.age_group = re.sub(r"\s*\(\d+\)", "", self.age_group).strip()

        # Extract sprint_time and sprint_points from e.g. "00:01:16 | 261 punkti"
        if self.sprint_distance:
            match = re.match(r"(\d+:\d+:\d+)\s*\|\s*(\d+)", self.sprint_distance)
            if match:
                self.sprint_time = datetime.strptime(match.group(1), "%H:%M:%S").time()
                self.sprint_points = int(match.group(2))
            self.sprint_distance = None

        # Validate points sum
        if self.points and self.sprint_points and self.total_points:
            expected = self.points + self.sprint_points
            if expected != self.total_points:
                raise ValueError(
                    f"Points mismatch for '{self.participant}': "
                    f"{self.points} + {self.sprint_points} = {expected}, but total_points = {self.total_points}"
                )

        return self


class StirnuBuks(BaseModel):
    results: List[StirnuBuksResult]
    year: int
    name: str
    location: str
    event_no: int
    no_events_season: int
    distance_name: Literal["vāvere", "zaķis", "stirnu_buks", "lūsis"]
    distance_km: float
    distance_km_actual: Optional[float]  # Sometimes race distance differs

    @field_validator("distance_name", mode="before")
    @classmethod
    def parse_distance_name(cls, v: str):
        return v.strip().lower().replace(" ", "_")


def parse_stirnu_buks_xml(xml: Union[Path, str]):
    if isinstance(xml, Path):
        tree = ET.parse(str(xml))
    elif isinstance(xml, str):
        tree = ET.ElementTree(ET.fromstring(xml))

    root = tree.getroot()
    rows = root.findall(".//row")

    # Extract headers
    headers = [cell.text.strip() for cell in rows[0].findall("cell")]

    # Extract data rows
    data = []
    for row in rows[1:]:
        row_data = []
        for cell in row.findall("cell"):
            # Get text content, including nested <p> tags
            parts = []
            if cell.text:
                parts.append(cell.text.strip())
            for child in cell:
                if child.text:
                    parts.append(child.text.strip())
            row_data.append(" | ".join(filter(None, parts)))

        if len(row_data) == len(headers):
            data.append(dict(zip(headers, row_data)))
    return data


def parse_stirnu_buks_races(races: list[dict]) -> list[StirnuBuks]:
    races_sb = []
    for race in races:
        downloaded = tt.fetch_url(race["url"])
        xml_results = tt.extract(downloaded, output_format="xml")

        results = parse_stirnu_buks_xml(xml_results)
        metadata = race["metadata"]

        race = StirnuBuks(results=results, **metadata)
        races_sb.append(race)

    return races_sb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process race results data.")

    parser.add_argument(
        '--parse-all-races', 
        action='store_true', 
        help='If set, parses all races. Default is False.'
    )

    parser.add_argument(
        '--path', 
        type=str, 
        required=True, 
        help='The file path to the results (e.g., results.csv or logs/data.txt)'
    )

    args = parser.parse_args()
    file_path = Path(args.path).resolve()

    if args.parse_all_races:
        races = json.loads(file_path.read_text())
        races_sb_raw = races.get("stirnu_buks")
        races_sb = parse_stirnu_buks_races(races_sb_raw)
        race0 = races_sb[0]
        result_example = race0.results[0]
        print(f"{race0.year}-{race0.name}")
        print(result_example.model_dump())
    else:
        race = parse_stirnu_buks_xml(file_path)
        result_example = StirnuBuksResult(**race[0])
        print(result_example.model_dump())
