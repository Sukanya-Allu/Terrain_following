import streamlit as st
import xml.etree.ElementTree as ET
from geopy.distance import geodesic, distance
from geopy import Point
import numpy as np
import requests
import json
import os
import time
import io

# Streamlit page configuration
st.set_page_config(page_title="VTOL Path Planner", page_icon="✈️", layout="wide")

# ───────────────────────────────────────────────────────────────────────────────
# Helper functions from the original script (unchanged)
def calculate_bearing(p1, p2):
    lat1, lat2 = np.radians(p1[0]), np.radians(p2[0])
    dlon = np.radians(p2[1] - p1[1])
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def move_point_along_bearing(start, bearing, dist_m):
    origin = Point(start[0], start[1])
    dest = distance(meters=dist_m).destination(origin, bearing)
    return (dest.latitude, dest.longitude)

# Global elevation cache
elevation_cache = {}

def fetch_elevations(coords, config, neighbor_coords=None):
    elevs = []
    default_elevation = config.get("default_safe_alt", 1000)
    cached_coords = []
    to_fetch = []
    for lat, lon in coords:
        key = (round(lat, 6), round(lon, 6))
        if key in elevation_cache:
            cached_coords.append(elevation_cache[key])
        else:
            to_fetch.append((lat, lon))
    if to_fetch:
        min_lat, max_lat = min(lat for lat, _ in to_fetch) - 0.0045, max(lat for lat, _ in to_fetch) + 0.0045
        min_lon, max_lon = min(lon for _, lon in to_fetch) - 0.0045, max(lon for _, lon in to_fetch) + 0.0045
        grid_lats = np.linspace(min_lat, max_lat, 20)
        grid_lons = np.linspace(min_lon, max_lon, 20)
        grid_coords = [(lat, lon) for lat in grid_lats for lon in grid_lons]
        grid_coords.extend(to_fetch)
    else:
        grid_coords = []
    open_elev_url = 'https://api.open-elevation.com/api/v1/lookup'
    usgs_url = 'https://epqs.nationalmap.gov/v1/json'
    for i in range(0, len(grid_coords), 50):
        batch = grid_coords[i:i+50]
        locs = [{'latitude': lat, 'longitude': lon} for lat, lon in batch]
        attempt = 0
        success = False
        max_retries = 2
        start_time = time.time()
        while attempt < max_retries and not success and time.time() - start_time < 5:
            try:
                r = requests.post(open_elev_url, json={'locations': locs}, timeout=5)
                if r.status_code == 200:
                    results = r.json().get('results', [])
                    batch_elevs = [pt['elevation'] for pt in results]
                    if all(-500 <= e <= 8000 for e in batch_elevs):
                        elevs.extend(batch_elevs)
                        for (lat, lon), elev in zip(batch, batch_elevs):
                            elevation_cache[(round(lat, 6), round(lon, 6))] = elev
                        success = True
                    else:
                        raise ValueError("Invalid elevation data")
                else:
                    raise requests.RequestException("API error")
            except (requests.RequestException, ValueError):
                attempt += 1
                time.sleep(0.3 * attempt)
        if not success:
            for lat, lon in batch:
                attempt = 0
                usgs_success = False
                while attempt < max_retries and not usgs_success:
                    try:
                        params = {'x': lon, 'y': lat, 'units': 'Meters', 'output': 'json'}
                        r = requests.get(usgs_url, params=params, timeout=5)
                        if r.status_code == 200:
                            elev = r.json().get('value', default_elevation)
                            elevs.append(elev if elev != 'N/A' else default_elevation)
                            usgs_success = True
                        else:
                            raise requests.RequestException("USGS API error")
                    except:
                        attempt += 1
                        time.sleep(0.3 * attempt)
                    if not usgs_success:
                        # Use higher peak elevation from neighbor_coords if provided
                        if neighbor_coords:
                            prev_segment, next_segment = neighbor_coords
                            max_neighbor_elev = 0
                            if prev_segment:
                                prev_elevs = fetch_elevations([prev_segment] if isinstance(prev_segment, tuple) else prev_segment, config, None)
                                max_neighbor_elev = max(max_neighbor_elev, max(prev_elevs) if prev_elevs else 0)
                            if next_segment:
                                next_elevs = fetch_elevations([next_segment] if isinstance(next_segment, tuple) else next_segment, config, None)
                                max_neighbor_elev = max(max_neighbor_elev, max(next_elevs) if next_elevs else 0)
                            elevs.append(max_neighbor_elev + config["safety_margin"])
                        else:
                            elevs.append(default_elevation)
        time.sleep(0.1)
    final_elevs = []
    for lat, lon in coords:
        key = (round(lat, 6), round(lon, 6))
        if key in elevation_cache:
            final_elevs.append(elevation_cache[key])
        else:
            final_elevs.append(elevs.pop(0) if elevs else default_elevation)
    if elevs and any(e > -500 for e in  final_elevs):
        elevs = np.array(final_elevs)
        elevs = np.convolve(elevs, np.ones(3)/3, mode='valid')
        peak_elev = np.percentile(elevs, 95)
    else:
        # Use higher peak elevation from neighbor_coords if provided
        if neighbor_coords:
            prev_segment, next_segment = neighbor_coords
            max_neighbor_elev = 0
            if prev_segment:
                prev_elevs = fetch_elevations([prev_segment] if isinstance(prev_segment, tuple) else prev_segment, config, None)
                max_neighbor_elev = max(max_neighbor_elev, max(prev_elevs) if prev_elevs else 0)
            if next_segment:
                next_elevs = fetch_elevations([next_segment] if isinstance(next_segment, tuple) else next_segment, config, None)
                max_neighbor_elev = max(max_neighbor_elev, max(next_elevs) if next_elevs else 0)
            peak_elev = max_neighbor_elev if max_neighbor_elev > 0 else default_elevation
        else:
            peak_elev = default_elevation
    if peak_elev < 100:
        st.warning(f"⚠️ Peak elevation {peak_elev}m seems low. Using {default_elevation}m.")
        peak_elev = default_elevation
    return [peak_elev] * len(coords)

def generate_takeoff_and_cruise(home, first_survey_point, config):
    takeoff_alt1 = config["takeoff_altitudes"][0]
    climb_alt2 = config["takeoff_altitudes"][1]
    climb_alt3 = config["takeoff_altitudes"][2]
    d1, d3_offset = 320, 500
    safety_margin = config.get("safety_margin", 150)
    path = []
    trigger_points = []
    item_id_counter = 1
    brg = calculate_bearing(home, first_survey_point)
    wp1 = (home[0], home[1], takeoff_alt1)
    path.append(wp1)
    trigger_points.append({
        "lat": wp1[0], "lon": wp1[1], "alt": wp1[2],
        "trigger_type": "none", "trigger_params": {}
    })
    item_id_counter += 1
    wp2_xy = move_point_along_bearing(home, brg, d1)
    wp2 = (wp2_xy[0], wp2_xy[1], climb_alt2)
    path.append(wp2)
    trigger_points.append({
        "lat": wp2[0], "lon": wp2[1], "alt": wp2[2],
        "trigger_type": "none", "trigger_params": {}
    })
    item_id_counter += 1
    wp3_xy = move_point_along_bearing(wp2_xy, brg, d3_offset)
    wp3 = (wp3_xy[0], wp3_xy[1], climb_alt3)
    path.append(wp3)
    trigger_points.append({
        "lat": wp3[0], "lon": wp3[1], "alt": wp3[2],
        "trigger_type": "none", "trigger_params": {}
    })
    item_id_counter += 1
    entry = move_point_along_bearing(first_survey_point, (brg + 180) % 360, config["turning_length"])
    total_dist = geodesic(wp3_xy, entry).meters
    num_samples = max(10, int(total_dist / 50))
    brg_to_entry = calculate_bearing(wp3_xy, entry)
    sample_coords = []
    for i in range(num_samples + 1):
        dist = total_dist * (i / num_samples)
        intermediate = move_point_along_bearing(wp3_xy, brg_to_entry, dist)
        sample_coords.append(intermediate)
    elevs = fetch_elevations(sample_coords,config=config, neighbor_coords=(None, [first_survey_point]))
    safe_alt = elevs[0] + safety_margin
    wp4 = (wp3_xy[0], wp3_xy[1], safe_alt)
    path.append(wp4)
    trigger_points.append({
        "lat": wp4[0], "lon": wp4[1], "alt": wp4[2],
        "trigger_type": "loiter", "trigger_params": {"radius": config["loiter_radius"]}
    })
    item_id_counter += 1
    wp7_xy = move_point_along_bearing(entry, (brg_to_entry + 180) % 360,500)
    d_total = geodesic(wp4[:2], wp7_xy).meters
    dists = np.linspace(0, d_total, 4)[1:3]
    wp5_xy = move_point_along_bearing(wp4[:2], brg_to_entry, dists[0])
    wp6_xy = move_point_along_bearing(wp4[:2], brg_to_entry, dists[1])
    wp5 = (wp5_xy[0], wp5_xy[1], safe_alt)
    wp6 = (wp6_xy[0], wp6_xy[1], safe_alt)
    path.extend([wp5, wp6])
    trigger_points.extend([
        {"lat": wp5[0], "lon": wp5[1], "alt": wp5[2], "trigger_type": "none", "trigger_params": {}},
        {"lat": wp6[0], "lon": wp6[1], "alt": wp6[2], "trigger_type": "none", "trigger_params": {}}
    ])
    item_id_counter += 2
    # Sample path from Loiter 1 (wp4) to mission start (entry) for elevation
    total_dist_entry = geodesic(wp4[:2], first_survey_point).meters
    num_samples_entry = max(20, int(total_dist_entry / 25))
    sample_coords_entry = []
    for i in range(num_samples_entry + 1):
        dist = total_dist_entry * (i / num_samples_entry)
        intermediate = move_point_along_bearing(wp4[:2],calculate_bearing(wp4[:2], first_survey_point), dist)
        sample_coords_entry.append(intermediate)
    # Fetch elevations with neighbor fallback (no default elevation)
    elevations = fetch_elevations(sample_coords_entry, config=config, neighbor_coords=(
        path[-2][:2] if len(path) >= 2 else None,  # Left neighbor
        [first_survey_point]  # Right neighbor
    ))
    if not elevations or all(e <= 0 for e in elevations):
        # Use neighbor elevations if fetch fails
        prev_segment, next_segment = (
            path[-2][:2] if len(path) >= 2 else None,
            [first_survey_point]
        )
        max_neighbor_elev = 0
        if prev_segment:
            prev_elevs = fetch_elevations([prev_segment], config, None)
            max_neighbor_elev = max(max_neighbor_elev, max(prev_elevs) if prev_elevs else 0)
        if next_segment:
            next_elevs = fetch_elevations(next_segment, config, None)
            max_neighbor_elev = max(max_neighbor_elev, max(next_elevs) if next_elevs else 0)
        cruise_alt = max_neighbor_elev + config["safety_margin"]
    else:
        cruise_alt = max(elevations) + config["safety_margin"]   #peak + 150m
    wp7 = (wp7_xy[0], wp7_xy[1], cruise_alt)
    path.append(wp7)
    trigger_points.append({
        "lat": wp7[0], "lon": wp7[1], "alt": wp7[2],
        "trigger_type": "loiter", "trigger_params": {"radius": config["loiter_radius"]}
    })
    item_id_counter += 1
    return path, trigger_points
def mission_end_to_rtl(last_point, last_alt, rtl_pt, config):
    """
    Generate path from last survey point to RTL with terrain-aware altitudes.
    Returns: list of (lat, lon, alt) points with loiter at specified distance.
    """
    rtl_points = []
    last_xy = (last_point[0], last_point[1])
    loiter_distance = config.get("loiter_distance", 500)
    safety_margin = config.get("safety_margin", 150)
    
    # Calculate loiter point
    brg_to_rtl = calculate_bearing(last_xy, rtl_pt)
    loiter_xy = move_point_along_bearing(rtl_pt, (brg_to_rtl + 180) % 360, loiter_distance)
    
    # Dense sampling (every ~50m)
    total_dist = geodesic(last_xy, loiter_xy).meters
    num_samples = max(5, int(total_dist / 50))
    sample_coords = []
    for i in range(num_samples + 1):
        dist = total_dist * (i / num_samples)
        intermediate = move_point_along_bearing(last_xy, brg_to_rtl, dist)
        sample_coords.append(intermediate)
    sample_coords.append(rtl_pt)  # Include RTL point
    
    # Fetch peak elevation
    elevs = fetch_elevations(sample_coords, config=config,neighbor_coords=(config.get("main_lines", [None])[-1] if config.get("main_lines") else None, None))
    safe_alt = elevs[0] + safety_margin  # Peak + 150m
    
    # Generate 5 points
    lats = np.linspace(last_xy[0], loiter_xy[0], 4).tolist() + [rtl_pt[0]]
    lons = np.linspace(last_xy[1], loiter_xy[1], 4).tolist() + [rtl_pt[1]]
    for i in range(5):
        lat, lon = lats[i], lons[i]
        if i == 4:
            # RTL: Fixed at 40m to match takeoff altitude
            alt = 40
        else:
            alt = max(last_alt, safe_alt)  # Use highest of survey or RTL path altitude
        rtl_points.append((lat, lon, alt))
    
    return rtl_points

def parse_kml(kml_content):
    try:
        tree = ET.ElementTree(ET.fromstring(kml_content))
        root = tree.getroot()
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        coords = []
        for ls in root.findall('.//kml:LineString', ns):
            for c in ls.find('kml:coordinates', ns).text.strip().split():
                lon, lat, *_ = map(float, c.split(','))
                coords.append((lat, lon))
        return coords
    except ET.ParseError:
        st.error("❌ Invalid KML format.")
        return []
    except Exception:
        st.error("❌ Failed to parse KML coordinates.")
        return []
def calculate_angle(p1, p2, p3):
    a = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    b = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 180.0
    return np.degrees(np.arccos(np.clip(np.dot(a, b)/(na*nb), -1,1)))

def split_lines_by_turn(coords, thresh=170):
    segments, cur = [], [coords[0]]
    for i in range(1, len(coords)-1):
        cur.append(coords[i])
        if calculate_angle(coords[i-1], coords[i], coords[i+1]) < thresh:
            segments.append(cur)
            cur = [coords[i]]
    cur.append(coords[-1])
    segments.append(cur)
    return segments

def filter_main_lines(lines, min_len=500):
    out = []
    for seg in lines:
        d = sum(geodesic(seg[i],seg[i+1]).meters for i in range(len(seg)-1))
        if d >= min_len and len(seg) >= 2:
            out.append(seg)
    return out

def adjust_line_directions(lines):
    if len(lines) > 1:
        for i in range(1, len(lines)):
            prev_end = lines[i-1][-1]
            curr_start = lines[i][0]
            curr_end = lines[i][-1]
            if geodesic(prev_end, curr_start).meters > geodesic(prev_end, curr_end).meters:
                lines[i].reverse()
    return lines

def create_trigger_item(lat, lon, alt, trigger_type, trigger_distance, item_id):
    if trigger_type == "camera":
        return {
            "AMSLAltAboveTerrain": None,
            "Altitude": alt,
            "AltitudeMode": 1,
            "autoContinue": True,
            "command": 206,
            "doJumpId": item_id,
            "frame": 3,
            "params": [
                trigger_distance,
                0,
                1,
                1,
                0, 0, 0
            ],
            "type": "SimpleItem"
        }
    return None
def generate_simplified_path(lines, home_pt, rtl_pt, config):
    path = []
    trigger_points = []
    item_id_counter = 1
    first_survey = lines[0][0]
    init_points, init_triggers = generate_takeoff_and_cruise(home_pt, first_survey, config)
    path.extend(init_points)
    trigger_points.extend(init_triggers)
    item_id_counter += len(init_points)
    lines = adjust_line_directions(lines)
    prev_exit, prev_alt = None, path[-1][2]
    total_segments = len(lines)
    for i, seg in enumerate(lines):
        st.text(f"Processing segment {i+1}/{total_segments}...")
        total_seg_dist = sum(geodesic(seg[j], seg[j+1]).meters for j in range(len(seg)-1))
        num_seg_samples = max(5, int(total_seg_dist / 50))
        seg_coords = []
        for j in range(len(seg)-1):
            p1, p2 = seg[j], seg[j+1]
            seg_dist = geodesic(p1, p2).meters
            samples = max(3, int(seg_dist / 50))
            lats = np.linspace(p1[0], p2[0], samples)
            lons = np.linspace(p1[1], p2[1], samples)
            seg_coords.extend(list(zip(lats, lons)))
        ev = fetch_elevations(seg_coords, config=config,neighbor_coords=(
            lines[i-1] if i > 0 else None,
            lines[i+1] if i < len(lines)-1 else None))
        cruise = ev[0] + config["safety_margin"]
        a, b = seg[0], seg[-1]
        brg = calculate_bearing(a, b)
        entry = move_point_along_bearing(a, (brg + 180) % 360, config["turning_length"])
        exit_pt = move_point_along_bearing(b, brg, 330)
        if i == 0:
            path.append((entry[0], entry[1], cruise))
            trigger_points.append({
                "lat": entry[0], "lon": entry[1], "alt": cruise,
                "trigger_type": "none",
                "trigger_params": {}
            })
            item_id_counter += 1
        elif prev_exit:
            path.append((prev_exit[0], prev_exit[1], prev_alt))
            trigger_points.append({
                "lat": prev_exit[0], "lon": prev_exit[1], "alt": prev_alt,
                "trigger_type": "none",
                "trigger_params": {}
            })
            item_id_counter += 1
            path.append((entry[0], entry[1], cruise))
            trigger_points.append({
                "lat": entry[0], "lon": entry[1], "alt": cruise,
                "trigger_type": "none",
                "trigger_params": {}
            })
            item_id_counter += 1
        path.extend([
            (entry[0], entry[1], cruise),
            (a[0], a[1], cruise),
            (b[0], b[1], cruise),
        ])
        trigger_points.extend([
            {
                "lat": entry[0], "lon": entry[1], "alt": cruise,
                "trigger_type": "none",
                "trigger_params": {}
            },
            {
                "lat": a[0], "lon": a[1], "alt": cruise,
                "trigger_type": "camera",
                "trigger_params": {"distance": config["trigger_distance"]}
            },
            {
                "lat": b[0], "lon": b[1], "alt": cruise,
                "trigger_type": "camera",
                "trigger_params": {"distance": config["end_trigger_distance"]}
            },
        ])
        item_id_counter += 3
        if i == len(lines) - 1:
            path.append((exit_pt[0], exit_pt[1], cruise))
            trigger_points.append({
                "lat": exit_pt[0], "lon": exit_pt[1], "alt": cruise,
                "trigger_type": "loiter",
                "trigger_params": {"radius": config["loiter_radius"]}
            })
            item_id_counter += 1
        else:
            path.append((exit_pt[0], exit_pt[1], cruise))
            trigger_points.append({
                "lat": exit_pt[0], "lon": exit_pt[1], "alt": cruise,
                "trigger_type": "none",
                "trigger_params": {}
            })
            item_id_counter += 1
            prev_exit, prev_alt = exit_pt, cruise
    st.text("Generating RTL path...")
    rtl_points = mission_end_to_rtl(exit_pt, cruise, rtl_pt, config)  # Fixed call
    path.extend(rtl_points)
    for i, pt in enumerate(rtl_points):
        trigger_type = "loiter" if i == 3 else "none"
        trigger_points.append({
            "lat": pt[0], "lon": pt[1], "alt": pt[2],
            "trigger_type": trigger_type,
            "trigger_params": {"radius": config["loiter_radius"]} if trigger_type == "loiter" else {}
        })
        item_id_counter += 1
    return path, trigger_points
def write_qgc_plan(points, trigger_points, start_trigger_distance, end_trigger_distance):
    items = []
    item_id = 1
    items.append({
        "Altitude": points[0][2],
        "AMSLAltAboveTerrain": points[0][2],
        "AltitudeMode": 1,
        "autoContinue": True,
        "command": 84,
        "doJumpId": item_id,
        "frame": 3,
        "params": [
            points[0][2], 0, 0, None,
            points[0][0], points[0][1], points[0][2]
        ],
        "type": "SimpleItem"
    })
    item_id += 1
    for i, (lat, lon, alt) in enumerate(points[1:], start=1):
        trigger = trigger_points[i]
        if trigger["trigger_type"] == "loiter":
            radius = trigger["trigger_params"].get("radius", 200)
            items.append({
                "Altitude": alt,
                "AMSLAltAboveTerrain": alt,
                "AltitudeMode": 1,
                "autoContinue": True,
                "command": 31,
                "doJumpId": item_id,
                "frame": 3,
                "params": [
                    1, radius, 0, 1, lat, lon, alt
                ],
                "type": "SimpleItem"
            })
            item_id += 1
        else:
            items.append({
                "Altitude": alt,
                "AMSLAltAboveTerrain": alt,
                "AltitudeMode": 1,
                "autoContinue": True,
                "command": 16,
                "doJumpId": item_id,
                "frame": 3,
                "params": [0, 0, 0, None, lat, lon, alt],
                "type": "SimpleItem"
            })
            item_id += 1
            if trigger["trigger_type"] == "camera":
                distance = trigger["trigger_params"].get("distance", start_trigger_distance)
                trigger_item = create_trigger_item(
                    trigger["lat"], trigger["lon"], trigger["alt"],
                    trigger["trigger_type"], distance, item_id
                )
                if trigger_item:
                    items.append(trigger_item)
                    item_id += 1
    items.append({
        "Altitude": 0,
        "AMSLAltAboveTerrain": 0,
        "AltitudeMode": 1,
        "autoContinue": True,
        "command": 20,
        "doJumpId": item_id,
        "frame": 3,
        "params": [0, 0, 0, 0, 0, 0, 0],
        "type": "SimpleItem"
    })
    plan = {
        "fileType": "Plan",
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "groundStation": "QGroundControl",
        "mission": {
            "items": items,
            "plannedHomePosition": [points[0][0], points[0][1], points[0][2]],
            "cruiseSpeed": 15,
            "hoverSpeed": 5,
            "firmwareType": 3,
            "vehicleType": 1,
            "version": 2,
            "globalPlanSettings": {
                "turnAroundDistance": 50,
                "useSafeTurnArcs": True,
                "useSplineWaypoints": True
            }
        },
        "rallyPoints": {"points": [], "version": 2},
        "version": 1
    }
    # Save to file and return JSON for download
    output_file = "mission.plan"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2)
    return plan
# Streamlit UI
st.title("Terrain Following Plan for QGroundControl")
st.markdown("Upload a KML file and provide mission parameters to generate a QGroundControl mission plan.")

# Input form
with st.form("mission_form"):
    st.subheader("Mission Parameters")
    col1, col2 = st.columns(2)
    with col1:
        home_lat = st.number_input("Home Latitude (used for RTL)", value=0.0, format="%.6f", step=0.000001)
        home_lon = st.number_input("Home Longitude (used for RTL)", value=0.0, format="%.6f", step=0.000001)
    with col2:
        st.markdown("**RTL Point**")
        st.info("RTL uses the same coordinates as the Home Point. Altitude fixed at 40m to match takeoff.")
    st.number_input("3rd WP Altitude (fixed at 150m, ignored)", value=150.0, disabled=True)
    start_trigger = st.number_input("Start Camera Trigger Distance (meters)", value=40.0, min_value=0.0, step=1.0)
    end_trigger = st.number_input("End Camera Trigger Distance (meters)", value=0.0, min_value=0.0, step=1.0)
    kml_file = st.file_uploader("Upload KML File", type=["kml"])
    submit = st.form_submit_button("Generate Mission Plan")

# Process form submission
if submit:
    if kml_file is None:
        st.error("❌ Please upload a KML file.")
    elif not (-90 <= home_lat <= 90 and -180 <= home_lon <= 180):
        st.error("❌ Invalid Home Latitude or Longitude.")
    else:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Initializing mission plan...")
            
            config = {
                "takeoff_altitudes": [40, 60, 150],
                "loiter_radius": 200,
                "loiter_distance": 500,
                "safety_margin": 150,
                "default_safe_alt": 1000,
                "trigger_distance": start_trigger,
                "end_trigger_distance": end_trigger,
                "turning_length": 320,
                "output_filename": "mission.plan"
            }
            home_pt = (home_lat, home_lon)
            rtl_pt = (home_lat, home_lon)
            
            status_text.text("Parsing KML file...")
            progress_bar.progress(20)
            kml_content = kml_file.read().decode('utf-8')
            coords = parse_kml(kml_content)
            if not coords:
                st.error("❌ No valid LineString found in KML file.")
                progress_bar.empty()
                status_text.empty()
            else:
                status_text.text("Processing survey lines...")
                progress_bar.progress(40)
                segments = split_lines_by_turn(coords)
                main_lines = filter_main_lines(segments)
                config["main_lines"] = main_lines
                if not main_lines:
                    st.error("❌ No valid survey lines found after filtering.")
                    progress_bar.empty()
                    status_text.empty()
                else:
                    status_text.text("Generating flight path (may take 1–3 minutes)...")
                    progress_bar.progress(60)
                    path, trigger_points = generate_simplified_path(main_lines, home_pt, rtl_pt, config)
                    
                    status_text.text("Writing mission plan...")
                    progress_bar.progress(80)
                    plan = write_qgc_plan(path, trigger_points, config["trigger_distance"], config["end_trigger_distance"])
                    plan_json = json.dumps(plan, indent=2)
                    status_text.text("Finalizing plan...")
                    progress_bar.progress(100)
                    st.success("✅ Mission plan generated successfully!")
                    st.download_button(
                        label="Download Mission Plan",
                        data=plan_json,
                        file_name="mission.plan",
                        mime="application/json"
                    )
        except Exception as e:
            st.error(f"❌ Error processing: {str(e)}. Check KML file or internet connection.")
            progress_bar.empty()
            status_text.empty()
