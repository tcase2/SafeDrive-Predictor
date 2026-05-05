"""
Safety Recommendations engine — US Imperial units.
All thresholds use: °F, in/hr, miles, mph.
"""

from typing import List, Dict


def _rec(priority: str, icon: str, title: str, body: str) -> Dict[str, str]:
    return {"priority": priority, "icon": icon, "title": title, "body": body}


def generate_recommendations(
    risk_score: float,
    precipitation: float,   # in/hr
    visibility: float,      # miles
    temperature: float,     # °F
    hour_of_day: int,
    accidents: int,
    tickets: int,
    has_recall: bool,
    recalls: List[dict] | None = None,
    vehicle_age: int = 5,
    wind_speed: float = 0.0,  # mph
) -> List[Dict[str, str]]:
    """Return an ordered list of recommendation dicts (highest priority first)."""
    recs: List[Dict[str, str]] = []

    # ── Recall ──────────────────────────────────────────────────────────────
    if has_recall:
        components = ", ".join(r.get("component", "") for r in (recalls or [])[:3])
        detail = f" Affected components: {components}." if components else ""
        recs.append(_rec(
            "CRITICAL", "🚨",
            "Active Safety Recall Detected",
            f"Your vehicle has an open NHTSA safety recall.{detail} "
            "Visit an authorized dealer immediately for a FREE repair. "
            "Driving with an unresolved recall may be illegal in some states.",
        ))

    # ── Precipitation ────────────────────────────────────────────────────────
    if precipitation > 0.75:          # ~0.75 in/hr = heavy rain / near-flood
        recs.append(_rec(
            "HIGH", "🌧️",
            "Severe Precipitation — Consider Postponing",
            f"Current precipitation is {precipitation:.2f} in/hr — dangerous hydroplaning conditions. "
            "If you must drive, reduce speed by at least 30 mph and double your following distance.",
        ))
    elif precipitation > 0.2:         # 0.2 in/hr = moderate rain
        recs.append(_rec(
            "MEDIUM", "🌦️",
            "Wet Roads — Increase Following Distance",
            f"Precipitation of {precipitation:.2f} in/hr reduces tire grip significantly. "
            "Reduce speed by 15 mph and increase your following distance to at least 6 seconds.",
        ))
    elif precipitation > 0:
        recs.append(_rec(
            "LOW", "🌂",
            "Light Rain — Turn On Headlights",
            "Light rain reduces visibility for other drivers. "
            "Turn on headlights and keep windshield washer fluid topped up.",
        ))

    # ── Visibility ───────────────────────────────────────────────────────────
    if visibility < 0.5:              # < 0.5 miles = dense fog / near-zero vis
        recs.append(_rec(
            "HIGH", "🌫️",
            "Extremely Low Visibility",
            f"Visibility is only {visibility:.1f} miles. Use fog lights, reduce speed by at least 25 mph, "
            "and avoid passing. If visibility drops below 0.2 miles, pull over safely.",
        ))
    elif visibility < 2.0:            # < 2 miles = reduced visibility
        recs.append(_rec(
            "MEDIUM", "🌁",
            "Reduced Visibility — Slow Down",
            f"Visibility is {visibility:.1f} miles. Reduce speed by 15 mph and activate headlights.",
        ))

    # ── Temperature / Ice ────────────────────────────────────────────────────
    if temperature <= 32 and precipitation > 0:
        recs.append(_rec(
            "HIGH", "🧊",
            "Black Ice Risk",
            f"Freezing temperatures ({temperature:.0f}°F) combined with precipitation create black ice. "
            "Reduce speed to 20 mph below the limit, avoid hard braking, "
            "and increase following distance to 10 seconds.",
        ))
    elif temperature <= 32:
        recs.append(_rec(
            "MEDIUM", "❄️",
            "Freezing Temperatures",
            f"Temperature is {temperature:.0f}°F. Bridges and overpasses freeze first. "
            "Check tire pressure (cold air reduces PSI) and allow extra stopping distance.",
        ))
    elif temperature >= 100:
        recs.append(_rec(
            "LOW", "🌡️",
            "Extreme Heat — Check Tires & Coolant",
            f"Temperature is {temperature:.0f}°F. High heat causes tire blowouts and engine overheating. "
            "Check tire pressure, coolant level, and carry extra water.",
        ))

    # ── Nighttime ────────────────────────────────────────────────────────────
    is_night = (hour_of_day < 6) or (hour_of_day >= 21)
    if is_night:
        recs.append(_rec(
            "MEDIUM", "🌙",
            "Nighttime Driving",
            "Nighttime increases crash risk by ~40%. Ensure headlights, taillights, and turn signals "
            "are all working. Stay alert for pedestrians and wildlife.",
        ))

    # ── Driver history ───────────────────────────────────────────────────────
    if accidents >= 2:
        recs.append(_rec(
            "HIGH", "⚠️",
            "High Accident History",
            f"Your record shows {accidents} accidents in the last 3 years. "
            "Consider a defensive driving refresher course — many insurers offer premium discounts for completion.",
        ))
    elif accidents == 1:
        recs.append(_rec(
            "LOW", "📋",
            "Prior Accident on Record",
            "You have 1 accident in the last 3 years. Stay extra alert and maintain safe following distances.",
        ))

    if tickets >= 3:
        recs.append(_rec(
            "HIGH", "🚔",
            "High Ticket Count — Risk of License Suspension",
            f"You have {tickets} tickets in 3 years. Many states suspend licenses at 3-6 violations. "
            "Strictly observe posted speed limits, especially in adverse weather.",
        ))
    elif tickets >= 1:
        recs.append(_rec(
            "LOW", "📌",
            "Traffic Violations on Record",
            f"{tickets} ticket(s) in 3 years elevates your risk profile. "
            "Consider using a dashcam and cruise control to avoid inadvertent speed violations.",
        ))

    # ── Wind ─────────────────────────────────────────────────────────────────
    if wind_speed >= 40:              # 40 mph = high wind advisory threshold
        recs.append(_rec(
            "HIGH", "💨",
            "High Wind Advisory",
            f"Wind speed is {wind_speed:.0f} mph. High-profile vehicles (SUVs, trucks, trailers) "
            "are especially vulnerable. Reduce speed and use both hands on the wheel.",
        ))
    elif wind_speed >= 25:
        recs.append(_rec(
            "MEDIUM", "💨",
            "Gusty Winds — Stay Alert",
            f"Wind speed is {wind_speed:.0f} mph. Watch for debris and be prepared for sudden gusts "
            "on bridges and open highways.",
        ))

    # ── Vehicle age ──────────────────────────────────────────────────────────
    if vehicle_age >= 15:
        recs.append(_rec(
            "LOW", "🔧",
            "Older Vehicle — Check Safety Systems",
            f"A {vehicle_age}-year-old vehicle may lack modern safety features (AEB, lane assist, etc.). "
            "Ensure brakes, tires, lights, and steering are inspected at least annually.",
        ))

    # ── General critical risk ────────────────────────────────────────────────
    if risk_score >= 75 and not any(r["priority"] == "CRITICAL" for r in recs):
        recs.append(_rec(
            "CRITICAL", "🛑",
            "Critical Risk Level — Consider Not Driving",
            f"Composite risk score is {risk_score:.0f}/100. "
            "Multiple high-risk factors are present simultaneously. "
            "If possible, delay travel until conditions improve.",
        ))
    elif risk_score >= 55:
        recs.append(_rec(
            "MEDIUM", "⚡",
            "Elevated Risk — Drive With Extra Caution",
            "Multiple risk factors are active. Plan your route in advance, "
            "inform someone of your trip, and keep your phone charged.",
        ))

    # Always add universal tip
    recs.append(_rec(
        "INFO", "ℹ️",
        "Universal Safety Tip",
        "Buckle up, put the phone down, and never drive impaired. "
        "These three habits eliminate roughly 60% of fatal crash risk regardless of conditions.",
    ))

    # Sort: CRITICAL → HIGH → MEDIUM → LOW → INFO
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    recs.sort(key=lambda r: order.get(r["priority"], 5))

    return recs
