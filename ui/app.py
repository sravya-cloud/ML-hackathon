
import streamlit as st
import requests
import folium
import streamlit.components.v1 as components

# API configuration
API_URL = "http://10.30.202.127:8000"

st.set_page_config(
    page_title="PathSnap Evaluator",
    page_icon="🗺️",
    layout="wide"
)

# ---------------- PAGE HEADER ----------------

st.title("🗺️ PathSnap - Route Prediction Evaluator")
st.caption("AI-powered route prediction and evaluation system")

# ---------------- API CONNECTION ----------------

st.subheader("API Connection")

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )
    health_response.raise_for_status()
    health = health_response.json()

    mode = health.get("mode", "unknown")

    if mode == "real":
        st.success("Connected to Real Model API")
    elif mode == "stub":
        st.warning("Connected, but API is running in STUB mode.")
    else:
        st.info(f"API connected. Mode: {mode}")

except requests.RequestException as e:
    st.error("Cannot connect to Sravya's API.")
    st.write("Check that both laptops are on the same Wi-Fi and that her server is running.")
    st.code(str(e))
    st.stop()

# ---------------- FETCH TRIPS ----------------

try:
    trips_response = requests.get(
        f"{API_URL}/trips",
        timeout=10
    )
    trips_response.raise_for_status()

    trips = trips_response.json().get("trips", [])

    if not trips:
        st.warning("No evaluator trips are currently available.")
        st.stop()

except (requests.RequestException, ValueError) as e:
    st.error("Unable to fetch evaluator trips.")
    st.code(str(e))
    st.stop()

# ---------------- TRIP SELECTION ----------------

st.divider()
st.subheader("Select Evaluator Trip")

selected_trip = st.selectbox(
    "Choose a Trip ID",
    trips
)

# ---------------- PREDICTION ----------------

if st.button("Predict Route", type="primary"):

    with st.spinner("Analysing trip and predicting route..."):

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={
                    "trip_id": selected_trip,
                    "include_map": False
                },
                timeout=60
            )

            response.raise_for_status()
            result = response.json()

        except requests.RequestException as e:
            st.error("Prediction request failed.")
            st.code(str(e))
            st.stop()

    # ---------------- RESULTS ----------------

    st.success("Prediction completed!")

    st.subheader(f"Results for Trip: {result.get('trip_id', selected_trip)}")

    metrics = result.get("metrics", {})

    st.markdown("### Evaluation Metrics")

    metric_names = [
        ("Top-1 Accuracy", "top1_accuracy"),
        ("Macro F1 Score", "macro_f1"),
        ("Route Continuity", "route_continuity"),
        ("Inference Time", "inference_seconds")
    ]

    cols = st.columns(4)

    for col, (label, key) in zip(cols, metric_names):
        value = metrics.get(key)

        if value is None:
            display_value = "N/A"
        elif key == "inference_seconds":
            display_value = f"{value:.3f} sec"
        elif isinstance(value, (int, float)):
            display_value = f"{value:.2%}"
        else:
            display_value = str(value)

        col.metric(label, display_value)

    model_name = metrics.get("model", "N/A")
    st.write(f"**Model:** {model_name}")

    st.write(
        "**Number of GPS Points:**",
        result.get("n_gps_points", "N/A")
    )

    with st.expander("View all evaluation metrics"):
        st.json(metrics)

    # ---------------- PREDICTED ROUTE ----------------

    st.divider()
    st.subheader("Predicted Route")

    route_ids = result.get("predicted_route_arc_ids", [])

    if route_ids:
        st.write(route_ids)
    else:
        st.info("No predicted route IDs returned.")

    # ---------------- ROUTE MAP ----------------

    st.subheader("Route Visualization")

    gps_points = result.get("gps_points_lonlat", [])
    predicted_points = result.get("predicted_route_lonlat", [])

    valid_points = [
        p for p in gps_points + predicted_points
        if isinstance(p, (list, tuple))
        and len(p) >= 2
        and p[0] is not None
        and p[1] is not None
    ]

    if valid_points:

        center = [
            sum(p[1] for p in valid_points) / len(valid_points),
            sum(p[0] for p in valid_points) / len(valid_points)
        ]

        route_map = folium.Map(
            location=center,
            zoom_start=14
        )

        if gps_points:
            folium.PolyLine(
                [(p[1], p[0]) for p in gps_points],
                color="blue",
                weight=4,
                tooltip="GPS Route"
            ).add_to(route_map)

        if predicted_points:
            folium.PolyLine(
                [(p[1], p[0]) for p in predicted_points],
                color="red",
                weight=5,
                tooltip="Predicted Route"
            ).add_to(route_map)

        folium.LayerControl().add_to(route_map)

        route_map.fit_bounds(
            [(p[1], p[0]) for p in valid_points]
        )

        components.html(
            route_map._repr_html_(),
            height=550,
            scrolling=True
        )

        st.caption("🔵 Blue: Actual GPS Route | 🔴 Red: Predicted Route")

    else:
        st.warning("No route coordinates were returned by the API.")

    # ---------------- RAW RESPONSE ----------------

    with st.expander("View complete API response"):
        st.json(result)

st.divider()
st.caption("PathSnap | Machine Learning Hackathon")
