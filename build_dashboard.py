#!/usr/bin/env python3
"""
Vessel Tracking Dashboard Generator
Author: Senior Python GIS & Data Visualization Engineer
Description: Reads a high-frequency EngineLink CSV log, cleans GPS coordinates, 
             performs coordinate smoothing, calculates voyage KPIs, and templates 
             an interactive, responsive, dark-themed HTML dashboard with Plotly.js.
"""

import os
import re
import sys
import json
import math
import webbrowser
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from jinja2 import Template

# Clean power, speed, RPM, and heading ranges
class VesselDataProcessor:
    """Handles parsing, cleaning, interpolating, and smoothing of high-frequency vessel data."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None
        self.column_mapping: Dict[str, str] = {}
        
    def load_data(self) -> pd.DataFrame:
        """Loads CSV data and automatically detects column names using regex keywords."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found at: {self.file_path}")
            
        print(f"[INFO] Reading CSV from: {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        self._detect_columns()
        return self.df
        
    def _detect_columns(self):
        """Helper to match CSV columns to known GIS and vessel parameters via keyword regex."""
        columns = self.df.columns.tolist()
        
        # Mapping rules based on keywords
        patterns = {
            'timestamp': [r'start-end\s*time', r'start_end', r'timestamp'],
            'latitude': [r'^latitude\b', r'lat'],
            'lat_dir': [r'\bns\b', r'n/s', r'lat_dir'],
            'longitude': [r'^longitude\b', r'lon', r'lng'],
            'lon_dir': [r'\bew\b', r'e/w', r'lon_dir'],
            'power': [r'shaft\s*power', r'm/e.*power', r'power'],
            'rpm': [r'shaft\s*rpm', r'm/e.*rpm', r'rpm'],
            'speed': [r'ship\s*speed', r'speed\s*over\s*ground', r'sog', r'speed'],
            'heading': [r'heading', r'course', r'cog']
        }
        
        for key, pattern_list in patterns.items():
            matched_col = None
            # Exclude direction fields from coordinates matching
            exclusions = ['ns', 'ew', 'dir'] if key in ['latitude', 'longitude'] else []
            
            for pattern in pattern_list:
                for col in columns:
                    col_clean = col.strip().lower()
                    if exclusions and any(ex in col_clean for ex in exclusions):
                        continue
                    if re.search(pattern, col_clean):
                        matched_col = col
                        break
                if matched_col:
                    break
            
            if matched_col:
                self.column_mapping[key] = matched_col
            else:
                # If optional or fallback logic is needed
                self.column_mapping[key] = None

        # Verify critical mappings
        critical_keys = ['timestamp', 'latitude', 'lat_dir', 'longitude', 'lon_dir', 'power', 'rpm', 'speed', 'heading']
        missing_keys = [k for k in critical_keys if self.column_mapping.get(k) is None]
        
        if missing_keys:
            print(f"[WARNING] Some column mappings could not be auto-detected: {missing_keys}")
            # Fallback to manual selection or error out
            for key in missing_keys:
                # Let's search columns more aggressively if we didn't find them
                for col in columns:
                    col_clean = col.strip().lower()
                    if key == 'latitude' and 'lat' in col_clean and 'ns' not in col_clean:
                        self.column_mapping[key] = col
                    elif key == 'longitude' and 'lon' in col_clean and 'ew' not in col_clean:
                        self.column_mapping[key] = col
                    elif key == 'lat_dir' and 'ns' in col_clean:
                        self.column_mapping[key] = col
                    elif key == 'lon_dir' and 'ew' in col_clean:
                        self.column_mapping[key] = col
                        
            # Recheck after aggressive search
            still_missing = [k for k in critical_keys if self.column_mapping.get(k) is None]
            if still_missing:
                raise ValueError(f"Failed to auto-detect columns: {still_missing}. Column names: {columns}")
        
        print("[SUCCESS] Auto-detected Column Mapping:")
        for k, v in self.column_mapping.items():
            print(f"  {k:12} -> '{v}'")

    @staticmethod
    def _parse_nmea_coord(val: Any, direction: Any) -> float:
        """Converts NMEA style DDMM.MMMM coordinates to Decimal Degrees."""
        if pd.isna(val) or pd.isna(direction):
            return np.nan
        try:
            val_str = str(val).strip()
            dir_str = str(direction).strip().upper()
            
            # Clean non-numeric characters from the NMEA number string
            val_str = re.sub(r'[^\d\.]', '', val_str)
            val_float = float(val_str)
            
            # DDMM.MMMM -> DD is degrees, MM.MMMM is minutes
            # For latitude, DD is 2 digits. For longitude, DDD is 3 digits.
            # Thus, we can separate using division by 100
            degrees = int(val_float // 100)
            minutes = val_float % 100
            decimal_degrees = degrees + (minutes / 60.0)
            
            if dir_str in ['S', 'W']:
                decimal_degrees = -decimal_degrees
                
            return decimal_degrees
        except Exception:
            return np.nan

    def process_data(self) -> pd.DataFrame:
        """Performs data preparation steps: conversions, filtering, sorting, interpolation, and smoothing."""
        if self.df is None:
            self.load_data()
            
        df = self.df.copy()
        
        t_col = self.column_mapping['timestamp']
        lat_col = self.column_mapping['latitude']
        lat_dir_col = self.column_mapping['lat_dir']
        lon_col = self.column_mapping['longitude']
        lon_dir_col = self.column_mapping['lon_dir']
        
        # 1. Convert Start-End Time to datetime
        print("[INFO] Converting timestamps to datetime...")
        df['datetime'] = pd.to_datetime(df[t_col], errors='coerce')
        
        # 2. Remove duplicate timestamps
        print("[INFO] Removing duplicate timestamps...")
        df = df.drop_duplicates(subset=['datetime'])
        
        # 3. Parse coordinates to Decimal Degrees
        print("[INFO] Parsing NMEA GPS coordinates...")
        df['lat_dec'] = df.apply(lambda r: self._parse_nmea_coord(r[lat_col], r[lat_dir_col]), axis=1)
        df['lon_dec'] = df.apply(lambda r: self._parse_nmea_coord(r[lon_col], r[lon_dir_col]), axis=1)
        
        # 4. Filter invalid coordinates (lat: [-90, 90], lon: [-180, 180])
        # Also remove exact 0.0 dropouts
        print("[INFO] Filtering invalid GPS coordinates...")
        df = df[
            (df['lat_dec'] >= -90.0) & (df['lat_dec'] <= 90.0) &
            (df['lon_dec'] >= -180.0) & (df['lon_dec'] <= 180.0) &
            (df['lat_dec'].abs() > 0.01) & (df['lon_dec'].abs() > 0.01)
        ]
        
        # 5. Sort chronologically
        print("[INFO] Sorting chronologically...")
        df = df.sort_values(by='datetime').reset_index(drop=True)
        
        # 6. Interpolate missing coordinates
        print("[INFO] Interpolating missing GPS coordinates...")
        df['lat_dec'] = df['lat_dec'].interpolate(method='linear').ffill().bfill()
        df['lon_dec'] = df['lon_dec'].interpolate(method='linear').ffill().bfill()
        
        # 7. Smooth GPS coordinates
        # Try Savitzky-Golay filter from SciPy, fallback to pandas rolling mean if error occurs
        print("[INFO] Smoothing GPS coordinates...")
        try:
            from scipy.signal import savgol_filter
            # Window length must be odd, positive, and less than df length
            window_length = 31
            if len(df) < window_length:
                window_length = len(df) if len(df) % 2 != 0 else len(df) - 1
            if window_length >= 5:
                df['lat_smooth'] = savgol_filter(df['lat_dec'], window_length, 3)
                df['lon_smooth'] = savgol_filter(df['lon_dec'], window_length, 3)
                print(f"  [SUCCESS] Coordinates smoothed with Savitzky-Golay (window={window_length}, order=3)")
            else:
                df['lat_smooth'] = df['lat_dec']
                df['lon_smooth'] = df['lon_dec']
        except Exception as e:
            print(f"  [WARNING] Savitzky-Golay failed: {e}. Falling back to Rolling Average smoothing.")
            df['lat_smooth'] = df['lat_dec'].rolling(window=15, min_periods=1, center=True).mean()
            df['lon_smooth'] = df['lon_dec'].rolling(window=15, min_periods=1, center=True).mean()
            
        # 8. Extract other metrics, casting to numeric and filling NaNs
        for key in ['power', 'rpm', 'speed', 'heading']:
            col_name = self.column_mapping[key]
            df[f'{key}_clean'] = pd.to_numeric(df[col_name], errors='coerce').interpolate(method='linear').ffill().bfill()
            
        # Normalize Shaft Power (and other metrics for coloring)
        print("[INFO] Normalizing parameters for coloring...")
        for key in ['power', 'rpm', 'speed', 'heading']:
            val_min = df[f'{key}_clean'].min()
            val_max = df[f'{key}_clean'].max()
            if val_max == val_min:
                df[f'{key}_norm'] = 0.0
            else:
                df[f'{key}_norm'] = (df[f'{key}_clean'] - val_min) / (val_max - val_min)
                
        self.df = df
        return df

    def compute_kpis(self) -> Dict[str, Any]:
        """Calculates voyage statistics (KPIs)."""
        if self.df is None:
            self.process_data()
            
        df = self.df
        
        # 1. Total Distance (sum of haversine distances between consecutive coordinates)
        lat_arr = df['lat_smooth'].values
        lon_arr = df['lon_smooth'].values
        
        lat1 = lat_arr[:-1]
        lon1 = lon_arr[:-1]
        lat2 = lat_arr[1:]
        lon2 = lon_arr[1:]
        
        # Outer space/Haversine formula mapping
        R = 6371.0  # Earth radius in km
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distances_km = R * c
        
        total_distance_km = float(np.sum(distances_km))
        total_distance_nm = total_distance_km / 1.852  # km to Nautical Miles
        
        # 2. Average Speed (in Knots)
        avg_speed = float(df['speed_clean'].mean())
        
        # 3. Maximum Power (in kW)
        max_power = float(df['power_clean'].max())
        
        # 4. Duration
        duration_sec = (df['datetime'].max() - df['datetime'].min()).total_seconds()
        duration_min = float(duration_sec / 60.0)
        
        # 5. Map zoom and center
        lat_min, lat_max = float(df['lat_smooth'].min()), float(df['lat_smooth'].max())
        lon_min, lon_max = float(df['lon_smooth'].min()), float(df['lon_smooth'].max())
        center_lat = (lat_min + lat_max) / 2.0
        center_lon = (lon_min + lon_max) / 2.0
        
        # Dynamic zoom level estimation based on bounding box size
        lat_diff = abs(lat_max - lat_min)
        lon_diff = abs(lon_max - lon_min)
        max_diff = max(lat_diff, lon_diff)
        
        zoom_level = 11.0
        if max_diff > 0:
            zoom_level = min(18.0, max(2.0, 11.5 - math.log2(max_diff)))
            
        kpis = {
            'total_distance_nm': round(total_distance_nm, 2),
            'total_distance_km': round(total_distance_km, 2),
            'avg_speed_kts': round(avg_speed, 2),
            'max_power_kw': round(max_power, 1),
            'duration_min': round(duration_min, 1),
            'center_lat': center_lat,
            'center_lon': center_lon,
            'zoom_level': round(zoom_level, 2)
        }
        
        print("[SUCCESS] Computed Voyage KPIs:")
        for k, v in kpis.items():
            print(f"  {k:20} -> {v}")
            
        return kpis


class DashboardGenerator:
    """Generates the interactive standalone HTML file using Plotly.js and a premium dark UI template."""
    
    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MarineLink | Vessel Performance Dashboard</title>
    
    <!-- Fonts and Icons -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <!-- Plotly.js CDN -->
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    
    <style>
        :root {
            --bg-base: #060b11;
            --bg-sidebar: #0b131e;
            --bg-card: #111d2c;
            --border-color: #1a2c3f;
            --text-primary: #e2e8f0;
            --text-secondary: #64748b;
            --text-dim: #475569;
            
            --accent-cyan: #00e5ff;
            --accent-green: #00e676;
            --accent-yellow: #ffd600;
            --accent-orange: #ff9100;
            --accent-red: #ff1744;
            --accent-purple: #e040fb;
            
            --panel-width: 380px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            user-select: none;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-base);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
            display: flex;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-sidebar);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-secondary);
        }

        /* Sidebar Container */
        .sidebar {
            width: var(--panel-width);
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 20px;
            z-index: 10;
            overflow-y: auto;
            flex-shrink: 0;
            box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5);
        }

        .sidebar-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 15px;
            margin-bottom: 20px;
        }

        .logo {
            font-size: 1.2rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .logo i {
            color: var(--accent-cyan);
            text-shadow: 0 0 10px rgba(0, 229, 255, 0.4);
        }

        .system-status {
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--accent-green);
            background-color: rgba(0, 230, 118, 0.1);
            padding: 4px 8px;
            border-radius: 12px;
            border: 1px solid rgba(0, 230, 118, 0.2);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .status-dot {
            width: 6px;
            height: 6px;
            background-color: var(--accent-green);
            border-radius: 50%;
        }

        .pulsing {
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0% { transform: scale(0.95); opacity: 0.5; }
            50% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 8px var(--accent-green); }
            100% { transform: scale(0.95); opacity: 0.5; }
        }

        /* Vessel State Header Card */
        .vessel-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .vessel-icon-wrapper {
            width: 48px;
            height: 48px;
            background-color: rgba(0, 229, 255, 0.1);
            border: 1px solid rgba(0, 229, 255, 0.2);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            color: var(--accent-cyan);
        }

        .vessel-meta {
            display: flex;
            flex-direction: column;
        }

        .vessel-name {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .vessel-status-text {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 2px;
        }

        /* KPI Dashboard Grid */
        .section-title {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-secondary);
            letter-spacing: 0.08em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .section-title i {
            color: var(--accent-cyan);
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }

        .kpi-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 12px;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 3px;
            height: 100%;
            background-color: var(--text-dim);
        }

        .kpi-card.power::before { background-color: var(--accent-red); }
        .kpi-card.rpm::before { background-color: var(--accent-orange); }
        .kpi-card.speed::before { background-color: var(--accent-green); }
        .kpi-card.heading::before { background-color: var(--accent-purple); }

        .kpi-label {
            font-size: 0.65rem;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .kpi-value-row {
            display: flex;
            align-items: baseline;
            margin-top: 4px;
        }

        .kpi-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f8fafc;
            font-family: 'Roboto Mono', monospace;
        }

        .kpi-unit {
            font-size: 0.7rem;
            color: var(--text-secondary);
            margin-left: 4px;
            font-weight: 500;
        }

        /* Voyage Summary Stats */
        .voyage-summary {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }

        .summary-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(26, 44, 63, 0.5);
            font-size: 0.8rem;
        }

        .summary-row:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }

        .summary-row:first-child {
            padding-top: 0;
        }

        .summary-label {
            color: var(--text-secondary);
            font-weight: 500;
        }

        .summary-value {
            font-weight: 600;
            font-family: 'Roboto Mono', monospace;
            color: var(--text-primary);
        }

        /* GPS Status Panel */
        .gps-panel {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px 15px;
            margin-bottom: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .gps-coord-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
        }

        .gps-coord-label {
            color: var(--text-secondary);
            font-weight: 500;
        }

        .gps-coord-val {
            font-family: 'Roboto Mono', monospace;
            font-weight: 600;
            color: var(--accent-cyan);
        }

        /* User Controls & Selects */
        .control-group {
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .control-label {
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--text-secondary);
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        select {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-primary);
            padding: 10px;
            font-size: 0.85rem;
            outline: none;
            cursor: pointer;
            transition: border-color 0.2s, box-shadow 0.2s;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
        }

        select:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 8px rgba(0, 229, 255, 0.2);
        }

        /* Main Workspace Container */
        .main-workspace {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            position: relative;
        }

        /* Map area */
        .map-container {
            flex-grow: 1;
            width: 100%;
            height: 100%;
            background-color: var(--bg-base);
            position: relative;
        }

        #map-plot {
            width: 100%;
            height: 100%;
        }

        /* Floating Overlay Controls on Map */
        .map-overlay-title {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(11, 19, 30, 0.85);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px 18px;
            z-index: 5;
            pointer-events: none;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }

        .map-overlay-title h1 {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: 0.02em;
        }

        .map-overlay-title p {
            font-size: 0.7rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        /* Bottom Animation Control Bar */
        .playback-bar {
            background-color: var(--bg-sidebar);
            border-top: 1px solid var(--border-color);
            padding: 15px 25px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            z-index: 10;
            box-shadow: 0 -10px 30px rgba(0, 0, 0, 0.5);
        }

        .slider-row {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .time-display {
            font-family: 'Roboto Mono', monospace;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-primary);
            min-width: 75px;
            text-align: center;
            background: rgba(26, 44, 63, 0.4);
            border: 1px solid var(--border-color);
            padding: 4px 8px;
            border-radius: 4px;
        }

        .scrubber-container {
            flex-grow: 1;
            position: relative;
            display: flex;
            align-items: center;
        }

        /* Range Input Scrubber */
        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: var(--border-color);
            outline: none;
            cursor: pointer;
            transition: background 0.3s;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--accent-cyan);
            border: 2px solid #ffffff;
            box-shadow: 0 0 8px var(--accent-cyan);
            cursor: pointer;
            transition: transform 0.1s, background-color 0.2s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
            background: #ffffff;
        }

        .controls-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .playback-buttons {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .btn {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 16px;
            font-size: 0.85rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn:hover {
            background-color: var(--border-color);
            border-color: var(--text-secondary);
        }

        .btn.btn-play {
            background-color: rgba(0, 230, 118, 0.1);
            border-color: rgba(0, 230, 118, 0.2);
            color: var(--accent-green);
        }

        .btn.btn-play:hover {
            background-color: var(--accent-green);
            color: var(--bg-base);
            box-shadow: 0 0 15px rgba(0, 230, 118, 0.4);
            border-color: var(--accent-green);
        }

        .btn.btn-pause {
            background-color: rgba(255, 145, 0, 0.1);
            border-color: rgba(255, 145, 0, 0.2);
            color: var(--accent-orange);
        }

        .btn.btn-pause:hover {
            background-color: var(--accent-orange);
            color: var(--bg-base);
            box-shadow: 0 0 15px rgba(255, 145, 0, 0.4);
            border-color: var(--accent-orange);
        }

        .speed-control {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .speed-control span {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        .speed-badge {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            padding: 6px 12px;
            border-radius: 4px;
            font-family: 'Roboto Mono', monospace;
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--accent-cyan);
            min-width: 60px;
            text-align: center;
        }

        /* Responsive adjustments */
        @media (max-width: 900px) {
            body {
                flex-direction: column;
            }
            .sidebar {
                width: 100%;
                height: 45vh;
                border-right: none;
                border-bottom: 1px solid var(--border-color);
            }
            .main-workspace {
                height: 55vh;
            }
        }
    </style>
</head>
<body>

    <!-- Sidebar performance console -->
    <div class="sidebar">
        
        <div class="sidebar-header">
            <div class="logo">
                <i class="fa-solid fa-anchor"></i> MARINE LINK
            </div>
            <div class="system-status">
                <span class="status-dot pulsing"></span> ACTIVE LOG
            </div>
        </div>
        
        <!-- Vessel Profile Card -->
        <div class="vessel-card">
            <div class="vessel-icon-wrapper">
                <i class="fa-solid fa-ship"></i>
            </div>
            <div class="vessel-meta">
                <span class="vessel-name">ENGLINK DATA VESSEL</span>
                <span class="vessel-status-text">ECDIS TRANSMISSION ON</span>
            </div>
        </div>
        
        <!-- Dynamic Side Panel KPIs -->
        <div class="section-title">
            <i class="fa-solid fa-gauge-high"></i> CURRENT ENGINE & TELEMETRY
        </div>
        
        <div class="kpi-grid">
            <div class="kpi-card power">
                <span class="kpi-label">SHAFT POWER</span>
                <div class="kpi-value-row">
                    <span class="kpi-value" id="kpi-power">0.0</span>
                    <span class="kpi-unit">kW</span>
                </div>
            </div>
            
            <div class="kpi-card rpm">
                <span class="kpi-label">SHAFT SPEED</span>
                <div class="kpi-value-row">
                    <span class="kpi-value" id="kpi-rpm">0.0</span>
                    <span class="kpi-unit">RPM</span>
                </div>
            </div>
            
            <div class="kpi-card speed">
                <span class="kpi-label">VESSEL SPEED</span>
                <div class="kpi-value-row">
                    <span class="kpi-value" id="kpi-speed">0.0</span>
                    <span class="kpi-unit">KT</span>
                </div>
            </div>
            
            <div class="kpi-card heading">
                <span class="kpi-label">HEADING</span>
                <div class="kpi-value-row">
                    <span class="kpi-value" id="kpi-heading">0.0</span>
                    <span class="kpi-unit">°</span>
                </div>
            </div>
        </div>
        
        <!-- GPS Status Card -->
        <div class="section-title">
            <i class="fa-solid fa-location-crosshairs"></i> ECDIS GEOMETRY
        </div>
        
        <div class="gps-panel">
            <div class="gps-coord-row">
                <span class="gps-coord-label">LATITUDE:</span>
                <span class="gps-coord-val" id="kpi-lat">--</span>
            </div>
            <div class="gps-coord-row">
                <span class="gps-coord-label">LONGITUDE:</span>
                <span class="gps-coord-val" id="kpi-lon">--</span>
            </div>
            <div class="gps-coord-row" style="border-top: 1px solid rgba(26,44,63,0.3); padding-top: 6px; margin-top: 4px;">
                <span class="gps-coord-label" style="color: var(--text-secondary)">UTC TIMETAG:</span>
                <span class="gps-coord-val" id="kpi-time" style="color: var(--text-primary)">--</span>
            </div>
        </div>
        
        <!-- Voyage Overall Summary KPIs -->
        <div class="section-title">
            <i class="fa-solid fa-chart-line"></i> VOYAGE TOTAL METRICS
        </div>
        
        <div class="voyage-summary">
            <div class="summary-row">
                <span class="summary-label">Total Distance</span>
                <span class="summary-value" id="summary-distance">{{ kpis.total_distance_nm }} NM ({{ kpis.total_distance_km }} km)</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Average Speed</span>
                <span class="summary-value" id="summary-avg-speed">{{ kpis.avg_speed_kts }} knots</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Max Shaft Power</span>
                <span class="summary-value" id="summary-max-power">{{ kpis.max_power_kw }} kW</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Voyage Duration</span>
                <span class="summary-value" id="summary-duration">{{ kpis.duration_min }} mins</span>
            </div>
        </div>
        
        <!-- Selection Controls -->
        <div class="section-title">
            <i class="fa-solid fa-sliders"></i> USER CONTROLS
        </div>
        
        <div class="control-group">
            <label class="control-label" for="color-by-select">Colour Route By</label>
            <select id="color-by-select">
                <option value="power" selected>M/E Shaft Power</option>
                <option value="rpm">M/E Shaft RPM</option>
                <option value="speed">Ship Speed</option>
                <option value="heading">Vessel Heading</option>
            </select>
        </div>
        
        <div class="control-group">
            <label class="control-label" for="map-style-select">Map Background</label>
            <select id="map-style-select">
                <option value="open-street-map" selected>OpenStreetMap (Default)</option>
                <option value="carto-darkmatter">Carto Darkmatter (Radar Style)</option>
                <option value="carto-positron">Carto Positron (Light Outline)</option>
            </select>
        </div>

    </div>

    <!-- Main Map workspace -->
    <div class="main-workspace">
        
        <!-- Map Overlay Header -->
        <div class="map-overlay-title">
            <h1>VESSEL VOYAGE TRAJECTORY</h1>
            <p>HIGH-FREQUENCY HARNESS PERFORMANCE MONITORING | FLIGHT FLUID PLOT</p>
        </div>
        
        <!-- Plotly Map target -->
        <div class="map-container">
            <div id="map-plot"></div>
        </div>
        
        <!-- Bottom Timeline playback controls -->
        <div class="playback-bar">
            
            <div class="slider-row">
                <span class="time-display" id="elapsed-time-display">00:00</span>
                <div class="scrubber-container">
                    <input type="range" id="time-slider" min="0" max="100" value="0">
                </div>
                <span class="time-display" id="total-time-display">00:00</span>
            </div>
            
            <div class="controls-row">
                <div class="playback-buttons">
                    <button class="btn btn-play" id="btn-play">
                        <i class="fa-solid fa-play"></i> PLAY
                    </button>
                    <button class="btn btn-pause" id="btn-pause" style="display: none;">
                        <i class="fa-solid fa-pause"></i> PAUSE
                    </button>
                    <button class="btn" id="btn-reset">
                        <i class="fa-solid fa-rotate-left"></i> RESTART
                    </button>
                </div>
                
                <div class="speed-control">
                    <span>Playback Speed:</span>
                    <button class="btn" id="btn-speed-down"><i class="fa-solid fa-minus"></i></button>
                    <span class="speed-badge" id="speed-badge">5x</span>
                    <button class="btn" id="btn-speed-up"><i class="fa-solid fa-plus"></i></button>
                </div>
            </div>
            
        </div>
        
    </div>

    <script>
        // Inject high-frequency data payload from Python
        const data = {
            timestamps: {{ timestamps_js }},
            lat: {{ lat_js }},
            lon: {{ lon_js }},
            power: {{ power_js }},
            rpm: {{ rpm_js }},
            speed: {{ speed_js }},
            heading: {{ heading_js }}
        };

        const ranges = {
            power: { min: {{ ranges.power.min }}, max: {{ ranges.power.max }}, title: 'M/E Shaft Power (kW)' },
            rpm: { min: {{ ranges.rpm.min }}, max: {{ ranges.rpm.max }}, title: 'M/E Shaft RPM' },
            speed: { min: {{ ranges.speed.min }}, max: {{ ranges.speed.max }}, title: 'Ship Speed (knots)' },
            heading: { min: {{ ranges.heading.min }}, max: {{ ranges.heading.max }}, title: 'Ship Heading (degrees)' }
        };

        const totalPoints = data.lat.length;
        
        // Define continuous custom color scale (Blue -> Green -> Yellow -> Orange -> Red)
        const customColorscale = [
            [0.0, '#00e5ff'],  // Cyan/Blue
            [0.25, '#00e676'], // Neon Green
            [0.5, '#ffd600'],  // Bright Yellow
            [0.75, '#ff9100'], // Orange
            [1.0, '#ff1744']   // Red
        ];

        // Format raw coordinates to degrees, minutes format (e.g. 24° 28.5316' N)
        function formatGPS(coord, isLat) {
            const dir = isLat ? (coord >= 0 ? 'N' : 'S') : (coord >= 0 ? 'E' : 'W');
            const absCoord = Math.abs(coord);
            const degrees = Math.floor(absCoord);
            const minutes = ((absCoord - degrees) * 60).toFixed(4);
            return `${degrees}° ${minutes}' ${dir}`;
        }

        // Format timestamps display timer
        function formatDuration(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }

        // Calculate a speed-based vector heading line
        function getHeadingVector(index) {
            const latStart = data.lat[index];
            const lonStart = data.lon[index];
            const speed = data.speed[index];
            const heading = data.heading[index];
            
            // Speed vector represents projected distance in 30 seconds
            // 30 seconds = 30 / 3600 hours = 1 / 120 hr
            // d_nm = speed * (30/3600)
            // d_deg = d_nm / 60 = speed * 30 / 216000
            const lengthDeg = (speed * 30.0) / 216000.0;
            const headingRad = (heading * Math.PI) / 180.0;
            
            const latEnd = latStart + lengthDeg * Math.cos(headingRad);
            // Adjust longitude vector length by latitude cosine factor
            const latCosine = Math.cos(latStart * Math.PI / 180.0);
            const lonEnd = lonStart + (lengthDeg * Math.sin(headingRad)) / (latCosine || 1.0);
            
            return {
                lat: [latStart, latEnd],
                lon: [lonStart, lonEnd]
            };
        }

        // Initialize scrubber range slider limits
        const timeSlider = document.getElementById('time-slider');
        timeSlider.max = totalPoints - 1;
        document.getElementById('total-time-display').innerText = formatDuration(totalPoints - 1);

        // State Machine Variables
        let currentIndex = 0;
        let isPlaying = false;
        let playbackSpeed = 5; // Default speed: skips 5 frames per animation loop
        let animationId = null;
        let lastFrameTime = 0;
        const targetFps = 30;
        const frameInterval = 1000 / targetFps; // ~33ms

        // Setup Traces for Plotly.js Map
        // Trace 0: Thin subtle base track line showing the whole voyage
        const traceBaseLine = {
            type: 'scattermapbox',
            lat: data.lat,
            lon: data.lon,
            mode: 'lines',
            line: {
                width: 1.5,
                color: '#1e293b'
            },
            hoverinfo: 'none',
            showlegend: false
        };

        // Trace 1: The colored points, using the selected metric
        const traceColoredMarkers = {
            type: 'scattermapbox',
            lat: data.lat,
            lon: data.lon,
            mode: 'markers',
            marker: {
                size: 3.5,
                color: data.power,
                colorscale: customColorscale,
                cmin: ranges.power.min,
                cmax: ranges.power.max,
                showscale: true,
                colorbar: {
                    title: {
                        text: 'M/E Shaft Power (kW)',
                        font: { color: '#e2e8f0', size: 11, family: 'Inter, sans-serif' }
                    },
                    tickfont: { color: '#94a3b8', size: 9, family: 'Roboto Mono, monospace' },
                    thickness: 16,
                    len: 0.55,
                    x: 0.98,
                    y: 0.5,
                    yanchor: 'middle',
                    bgcolor: 'rgba(11, 19, 30, 0.8)',
                    bordercolor: '#1a2c3f',
                    borderwidth: 1
                }
            },
            // Custom hover info showing complete sensor profile
            hovertemplate: 
                '<b>Time:</b> %{text}<br>' +
                '<b>Lat:</b> %{lat:.6f}°<br>' +
                '<b>Lon:</b> %{lon:.6f}°<br>' +
                '<b>Power:</b> %{customdata[0]:.1f} kW<br>' +
                '<b>RPM:</b> %{customdata[1]:.1f}<br>' +
                '<b>Speed:</b> %{customdata[2]:.2f} kt<br>' +
                '<b>Heading:</b> %{customdata[3]:.1f}°<br>' +
                '<extra></extra>',
            text: data.timestamps.map(t => t.split('T')[1].replace('Z', ' UTC')),
            customdata: data.power.map((p, i) => [p, data.rpm[i], data.speed[i], data.heading[i]]),
            showlegend: false
        };

        // Trace 2: Start & End Markers
        const traceStartEnd = {
            type: 'scattermapbox',
            lat: [data.lat[0], data.lat[totalPoints - 1]],
            lon: [data.lon[0], data.lon[totalPoints - 1]],
            mode: 'markers+text',
            marker: {
                size: 11,
                color: ['#00e676', '#ff1744'],
                symbol: 'circle'
            },
            text: ['START', 'END'],
            textposition: 'top center',
            textfont: {
                color: '#f8fafc',
                size: 10,
                weight: 'bold',
                family: 'Inter, sans-serif'
            },
            hoverinfo: 'none',
            showlegend: false
        };

        // Trace 3: Active Vessel outer glowing aura
        const traceVesselOuter = {
            type: 'scattermapbox',
            lat: [data.lat[0]],
            lon: [data.lon[0]],
            mode: 'markers',
            marker: {
                size: 18,
                color: 'rgba(0, 229, 255, 0.4)',
                symbol: 'circle'
            },
            hoverinfo: 'none',
            showlegend: false
        };

        // Trace 4: Active Vessel inner center point
        const traceVesselInner = {
            type: 'scattermapbox',
            lat: [data.lat[0]],
            lon: [data.lon[0]],
            mode: 'markers',
            marker: {
                size: 8,
                color: '#ffffff',
                symbol: 'circle'
            },
            hoverinfo: 'none',
            showlegend: false
        };

        // Trace 5: Dynamic projected speed vector line
        const initVec = getHeadingVector(0);
        const traceHeadingVector = {
            type: 'scattermapbox',
            lat: initVec.lat,
            lon: initVec.lon,
            mode: 'lines',
            line: {
                width: 3.5,
                color: '#00e5ff'
            },
            hoverinfo: 'none',
            showlegend: false
        };

        const traces = [
            traceBaseLine, 
            traceColoredMarkers, 
            traceStartEnd, 
            traceVesselOuter, 
            traceVesselInner, 
            traceHeadingVector
        ];

        // Mapbox Layout definition
        const layout = {
            mapbox: {
                style: 'open-street-map',
                center: { lat: {{ center_lat }}, lon: {{ center_lon }} },
                zoom: {{ zoom_level }}
            },
            margin: { r: 0, t: 0, l: 0, b: 0 },
            paper_bgcolor: '#060b11',
            plot_bgcolor: '#060b11',
            showlegend: false
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['select2d', 'lasso2d']
        };

        // Render Initial Map
        Plotly.newPlot('map-plot', traces, layout, config);

        // Core animation updates for sidebar dashboard and map elements
        function updateFrame(index) {
            if (index < 0 || index >= totalPoints) return;
            
            currentIndex = index;
            
            // 1. Update timeline controls
            timeSlider.value = index;
            document.getElementById('elapsed-time-display').innerText = formatDuration(index);
            
            // 2. Update Sidebar telemetry cards (KPI values)
            document.getElementById('kpi-power').innerText = data.power[index].toFixed(1);
            document.getElementById('kpi-rpm').innerText = data.rpm[index].toFixed(1);
            document.getElementById('kpi-speed').innerText = data.speed[index].toFixed(2);
            document.getElementById('kpi-heading').innerText = data.heading[index].toFixed(1);
            
            // 3. Update coordinates and timestamp details
            document.getElementById('kpi-lat').innerText = formatGPS(data.lat[index], true);
            document.getElementById('kpi-lon').innerText = formatGPS(data.lon[index], false);
            
            const rawTime = data.timestamps[index];
            // Format time cleanly (YYYY-MM-DD HH:MM:SS UTC)
            const cleanTime = rawTime.replace('T', ' ').replace('Z', ' UTC');
            document.getElementById('kpi-time').innerText = cleanTime;
            
            // 4. Calculate new projected speed heading vector
            const vec = getHeadingVector(index);
            
            // 5. Restyle Plotly layers (Trace 3: Outer Vessel, Trace 4: Inner Vessel, Trace 5: Speed Vector)
            Plotly.restyle('map-plot', {
                lat: [[data.lat[index]]],
                lon: [[data.lon[index]]]
            }, [3, 4]);
            
            Plotly.restyle('map-plot', {
                lat: [vec.lat],
                lon: [vec.lon]
            }, [5]);
        }

        // Loop animation method throttled to 30 FPS
        function animate(timestamp) {
            if (!isPlaying) return;
            
            if (!lastFrameTime) lastFrameTime = timestamp;
            const elapsed = timestamp - lastFrameTime;
            
            if (elapsed >= frameInterval) {
                // Ensure target rate alignment
                lastFrameTime = timestamp - (elapsed % frameInterval);
                
                let nextIndex = currentIndex + playbackSpeed;
                if (nextIndex >= totalPoints) {
                    nextIndex = 0; // Infinite loop behavior
                }
                
                updateFrame(nextIndex);
            }
            
            animationId = requestAnimationFrame(animate);
        }

        // Play/Pause Action handling
        const btnPlay = document.getElementById('btn-play');
        const btnPause = document.getElementById('btn-pause');

        function startPlayback() {
            if (isPlaying) return;
            isPlaying = true;
            btnPlay.style.display = 'none';
            btnPause.style.display = 'inline-flex';
            lastFrameTime = 0; // Reset delta timer
            animationId = requestAnimationFrame(animate);
        }

        function stopPlayback() {
            if (!isPlaying) return;
            isPlaying = false;
            btnPlay.style.display = 'inline-flex';
            btnPause.style.display = 'none';
            if (animationId) {
                cancelAnimationFrame(animationId);
                animationId = null;
            }
        }

        btnPlay.addEventListener('click', startPlayback);
        btnPause.addEventListener('click', stopPlayback);
        
        document.getElementById('btn-reset').addEventListener('click', () => {
            stopPlayback();
            updateFrame(0);
        });

        // Timeline Slider user interactions
        timeSlider.addEventListener('input', (e) => {
            stopPlayback();
            updateFrame(parseInt(e.target.value, 10));
        });

        // Playback Multiplier control buttons (speed factor range [1x, 100x])
        const speedBadge = document.getElementById('speed-badge');
        const speeds = [1, 2, 5, 10, 20, 50, 100];
        let speedIdx = speeds.indexOf(playbackSpeed);

        function updateSpeedDisplay() {
            speedBadge.innerText = `${playbackSpeed}x`;
        }

        document.getElementById('btn-speed-up').addEventListener('click', () => {
            if (speedIdx < speeds.length - 1) {
                speedIdx++;
                playbackSpeed = speeds[speedIdx];
                updateSpeedDisplay();
            }
        });

        document.getElementById('btn-speed-down').addEventListener('click', () => {
            if (speedIdx > 0) {
                speedIdx--;
                playbackSpeed = speeds[speedIdx];
                updateSpeedDisplay();
            }
        });

        // Dropdown Recoloring event handler
        const colorBySelect = document.getElementById('color-by-select');
        colorBySelect.addEventListener('change', (e) => {
            const metric = e.target.value;
            let values = [];
            const r = ranges[metric];
            
            if (metric === 'power') values = data.power;
            else if (metric === 'rpm') values = data.rpm;
            else if (metric === 'speed') values = data.speed;
            else if (metric === 'heading') values = data.heading;
            
            // Recolor Trace 1 (the colored track marker trace) and update colorbar title/ranges
            Plotly.restyle('map-plot', {
                'marker.color': [values],
                'marker.cmin': [r.min],
                'marker.cmax': [r.max],
                'marker.colorbar.title.text': [r.title]
            }, [1]);
        });

        // Map Style Changer dropdown handler
        const mapStyleSelect = document.getElementById('map-style-select');
        mapStyleSelect.addEventListener('change', (e) => {
            const style = e.target.value;
            Plotly.relayout('map-plot', {
                'mapbox.style': style
            });
        });

        // Load initial state for frame index 0 on startup
        updateFrame(0);

    </script>
</body>
</html>
"""

    def __init__(self, df: pd.DataFrame, kpis: Dict[str, Any]):
        self.df = df
        self.kpis = kpis
        
    def generate(self, output_path: str):
        """Compiles arrays to JSON formats, builds template strings, and writes the output HTML file."""
        print(f"[INFO] Formatting arrays for HTML template rendering...")
        
        # Prepare list fields
        timestamps = self.df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%SZ').tolist()
        lat = self.df['lat_smooth'].round(6).tolist()
        lon = self.df['lon_smooth'].round(6).tolist()
        power = self.df['power_clean'].round(1).tolist()
        rpm = self.df['rpm_clean'].round(1).tolist()
        speed = self.df['speed_clean'].round(2).tolist()
        heading = self.df['heading_clean'].round(1).tolist()
        
        # Create metrics ranges
        ranges = {
            'power': {'min': min(power), 'max': max(power)},
            'rpm': {'min': min(rpm), 'max': max(rpm)},
            'speed': {'min': min(speed), 'max': max(speed)},
            'heading': {'min': min(heading), 'max': max(heading)}
        }
        
        # Format variables as JSON strings to be safely printed in HTML script tags
        template_ctx = {
            'kpis': self.kpis,
            'center_lat': self.kpis['center_lat'],
            'center_lon': self.kpis['center_lon'],
            'zoom_level': self.kpis['zoom_level'],
            'timestamps_js': json.dumps(timestamps),
            'lat_js': json.dumps(lat),
            'lon_js': json.dumps(lon),
            'power_js': json.dumps(power),
            'rpm_js': json.dumps(rpm),
            'speed_js': json.dumps(speed),
            'heading_js': json.dumps(heading),
            'ranges': ranges
        }
        
        print(f"[INFO] Compiling template using Jinja2...")
        t = Template(self.HTML_TEMPLATE)
        rendered_html = t.render(**template_ctx)
        
        print(f"[INFO] Writing HTML output file to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
            
        print(f"[SUCCESS] Dashboard generated successfully: {output_path}")


def main():
    """Main execution sequence."""
    print("======================================================================")
    print("              VESSEL PERFORMANCE TRACKING BUILD SYSTEM                ")
    print("======================================================================")
    
    # 1. Determine input path
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    default_downloads_path = r"C:\Users\Dell\Downloads\EngineLink_20260708061746.csv"
    local_workspace_path = "EngineLink_20260708061746.csv"
    
    csv_path = None
    if cli_arg:
        csv_path = cli_arg
    elif os.path.exists(local_workspace_path):
        csv_path = local_workspace_path
    elif os.path.exists(default_downloads_path):
        csv_path = default_downloads_path
    else:
        # Prompt if neither is found
        print("[ERROR] CSV log file not found. Checked default paths:")
        print(f"  1. CLI argument")
        print(f"  2. {os.path.abspath(local_workspace_path)}")
        print(f"  3. {default_downloads_path}")
        sys.exit(1)
        
    output_html_name = "vessel_tracker.html"
    output_html_path = os.path.abspath(output_html_name)
    
    try:
        # 2. Pipeline processing
        processor = VesselDataProcessor(csv_path)
        df_processed = processor.process_data()
        kpis = processor.compute_kpis()
        
        # 3. Generating HTML Dashboard
        generator = DashboardGenerator(df_processed, kpis)
        generator.generate(output_html_path)
        
        # 4. Open in browser automatically
        print(f"[INFO] Opening dashboard in system default browser...")
        # Normalizes file path to absolute URI for local HTML files
        url = f"file:///{output_html_path.replace(os.sep, '/')}"
        webbrowser.open(url)
        print("[SUCCESS] Operational. Program exiting.")
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Dashboard generation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
