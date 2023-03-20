import pandas as pd
import numpy as np
import pickle

def decode_quality(code):
    if code == 3: return "good"
    elif code == 2: return "moderate"
    elif code == 1: return "bad"
    return "spam"

def calculate_days_passed(date, from_day):
    timestamp = pd.to_datetime(date)

    days = (from_day - timestamp).dt.days.astype(np.int16)

    return days

def transform(record):
    df = record.copy()
    today = pd.Timestamp.now()

    df["last_conn_days"] = calculate_days_passed(df["last_conn_date"], today)
    df["last_seen_days"] = calculate_days_passed(df["last_seen_date"], today)
    
    df.drop(columns = ["last_conn_date", "last_seen_date"], inplace = True)
    return df

class QualityEstimator:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.lgbm_model = pickle.load(f)
        
    def predict(self, record) -> str:
        model_input = transform(record)
        predict = self.lgbm_model.predict(model_input)[0]
        
        return decode_quality(predict)
    
SORTED_QUALITY = ["spam", "bad", "moderate", "good"]
def calculate_quality(scores):
    def calculate_quality_for_row(row):
        blacklist_score, dynamic_score = row["blacklist_score"], row["dynamic_score"]
        if blacklist_score == 1:
            return "spam"
        if dynamic_score < 0.3:
            return "bad"
        elif dynamic_score >= 0.3 and dynamic_score < 0.6:
            return "moderate"

        return "good"
    quality = scores.apply(calculate_quality_for_row, axis = 1)\
        .astype("category").cat.reorder_categories(SORTED_QUALITY)
    return quality

def calculate_quality_code(scores):
    quality = calculate_quality(scores)
    return quality.cat.codes.rename("quality_cat_id")

from typing import List, Tuple
import folium
from geopy.geocoders import Nominatim
import pandas as pd
quality_color_map = {
    "good": "green",
    "moderate": "orange",
    "bad": "red",
    "spam": "black",
}
def cast_quality_v3(row) -> str:
    score = row["score_v3"]
    return cast_quality(score)

def cast_quality_v4(row) -> str:
    score = row["score_v4"]
    return cast_quality(score)

def cast_color_v4(row) -> str:
    score = row["score_v4"]
    return quality_color_map[cast_quality(score)]


def cast_color_v3(row) -> str:
    score = row["score_v3"]
    return quality_color_map[cast_quality(score)]

def cast_quality(score):
    if score != score or score is None:
        return "spam"
    if score < 0.3:
        return "bad"
    elif score >= 0.3 and score < 0.6:
        return "moderate"

    return "good"

def create_markers_map(df: pd.DataFrame, cast_color) -> folium.Map:
    """Creates a Folium map with markers for each row in the DataFrame."""
    # Define the map's center coordinates
    center_lat = df['lat'].mean()
    center_lng = df['lng'].mean()

    # Create the Folium map
    map_obj = folium.Map(location=[center_lat, center_lng], zoom_start=12)

    def add_marker(row: pd.Series) -> None:
        """Adds a marker to the map for a given row in the DataFrame."""
        popup_html = f"<b>name: {row['name']}</b>"
        display_columns = ["address", "connections_count", "score_v3", "score_v4"]
        for column in display_columns:
            popup_html += f"<br>{column}: {row[column]}"

        icon_color = cast_color(row)
        icon = folium.Icon(color=icon_color, icon='')

        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=popup_html,
            icon=icon,
        ).add_to(map_obj)

    # Apply the add_marker function to each row of the DataFrame using the apply method
    df.apply(add_marker, axis=1)

    return map_obj


def create_markers_map_colored(df: pd.DataFrame, color: pd.Series) -> folium.Map:
    """Creates a Folium map with markers for each row in the DataFrame."""
    # Define the map's center coordinates
    center_lat = df['lat'].mean()
    center_lng = df['lng'].mean()

    # Create the Folium map
    map_obj = folium.Map(location=[center_lat, center_lng], zoom_start=12)
    
    def add_marker(row: pd.Series) -> None:
        """Adds a marker to the map for a given row in the DataFrame."""
        popup_html = f"<b>name: {row['name']}</b>"
        display_columns = ["address", "connections_count", "score_v3", "score_v4"]
        for column in display_columns:
            popup_html += f"<br>{column}: {row[column]}"

        icon = folium.Icon(color=row["color"], icon='')

        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=popup_html,
            icon=icon,
        ).add_to(map_obj)
    df["color"] = color
    # Apply the add_marker function to each row of the DataFrame using the apply method
    df.apply(add_marker, axis=1)

    return map_obj