"""Predict if ONE new booking will be canceled (command: predict).

It loads the saved pipeline (no training), turns the booking details the user
types into the same columns the model was trained on, and applies the saved threshold.
"""
import json

import pandas as pd

from src.classification.common import DEPLOYED_MODEL
from src.classification.evaluate import load_artifact

# the questions asked in interactive mode: key, question, default, type
QUESTIONS = [
    ("lead_time", "Days between booking and arrival", 60, int),
    ("arrival_month", "Arrival month (1-12)", 7, int),
    ("weekend_nights", "Weekend nights (Sat/Sun)", 1, int),
    ("week_nights", "Week nights (Mon-Fri)", 2, int),
    ("adults", "Adults", 2, int),
    ("children", "Children", 0, int),
    ("babies", "Babies", 0, int),
    ("meal", "Meal (BB, FB, HB, SC)", "BB", str),
    ("country", "Country code (e.g. PRT, GBR, FRA)", "PRT", str),
    ("market_segment", "Market segment (Online TA, Offline TA/TO, Direct, Corporate, Groups, Complementary, Aviation)", "Online TA", str),
    ("deposit_type", "Deposit type (No Deposit, Non Refund, Refundable)", "No Deposit", str),
    ("agent", "Agent ID (or NULL if none)", "9", str),
    ("has_company", "Booked through a company? (0 or 1)", 0, int),
    ("customer_type", "Customer type (Transient, Transient-Party, Contract, Group)", "Transient", str),
    ("special_requests", "Number of special requests", 0, int),
    ("previous_bookings_not_canceled", "Previous bookings the guest did NOT cancel", 0, int),
]


def ask_user():
    """Ask the questions in the terminal. Press Enter to keep the default."""
    booking = {}
    for key, question, default, cast in QUESTIONS:
        answer = input(f"{question} [{default}]: ").strip()
        booking[key] = cast(answer) if answer else default
    return booking


def _set(row, column, value, columns):
    """Set a column only if the model has it."""
    if column in columns:
        row[column] = value
        return True
    return False


def build_row(booking, meta):
    """Turn the booking details into ONE row with the exact training columns."""
    columns = meta["feature_columns"]
    row = {c: 0 for c in columns}                      # start with all zeros

    nights = booking["weekend_nights"] + booking["week_nights"]
    guests = booking["adults"] + booking["children"] + booking["babies"]
    if booking["adults"] < 1:
        raise ValueError("A booking needs at least 1 adult.")
    if not 1 <= booking["arrival_month"] <= 12:
        raise ValueError("Arrival month must be between 1 and 12.")

    row["LeadTime"] = booking["lead_time"]
    row["TotalOfSpecialRequests"] = booking["special_requests"]
    row["TotalNights"] = nights
    row["WeekendOrNo"] = int(booking["weekend_nights"] > 0)
    row["IsZeroNights"] = int(nights == 0)
    row["total_guests"] = guests
    row["has_company"] = int(booking["has_company"] > 0)
    row["has_prev_not_canceled"] = int(booking["previous_bookings_not_canceled"] > 0)
    row[f"month_{booking['arrival_month']}"] = 1

    # one-hot groups: switch on the matching column. A value the model never saw stays all zeros.
    notes = []
    if not _set(row, f"Meal_{booking['meal']}", 1, columns):
        notes.append(f"meal '{booking['meal']}' was not seen in training")
    if not _set(row, f"MarketSegment_{booking['market_segment']}", 1, columns):
        notes.append(f"market segment '{booking['market_segment']}' was not seen in training")
    if not _set(row, f"DepositType_{booking['deposit_type']}", 1, columns):
        notes.append(f"deposit type '{booking['deposit_type']}' was not seen in training")

    country = booking["country"].strip().upper()
    if country == "CN":                                  # the data uses CHN for China
        country = "CHN"
    if not _set(row, f"Country_{country}", 1, columns):
        row["Country_infrequent_sklearn"] = 1          # rare countries share one column
        notes.append(f"country '{country}' is rare, so it is grouped with other rare countries")

    # Transient is the baseline: all three flags stay 0 for it
    customer = booking["customer_type"].strip()
    if customer == "Transient-Party":
        row["CustomerType_Transient_Party"] = 1
    elif customer in ("Contract", "Group"):
        row[f"CustomerType_{customer}"] = 1
    elif customer != "Transient":
        notes.append(f"customer type '{customer}' was not seen in training")

    # Agent stays as text: the saved pipeline target-encodes it itself
    agent = str(booking["agent"]).strip()
    if agent.endswith(".0"):
        agent = agent[:-2]
    known = set(meta["known_agents"])
    if agent.upper() in ("", "NULL", "NONE", "NAN", "NO_AGENT"):
        agent = next((a for a in ("No_Agent", "NULL", "nan") if a in known), "No_Agent")
    elif agent not in known:
        notes.append(f"agent '{agent}' is new, so the model uses the average agent")
    row["Agent"] = agent

    return pd.DataFrame([row])[columns], notes         # columns in the training order


def predict_booking(booking, model_key=DEPLOYED_MODEL):
    pipe, meta = load_artifact(model_key)
    row, notes = build_row(booking, meta)
    proba = float(pipe.predict_proba(row)[0, 1])
    return {"model": meta["label"], "probability_of_cancel": round(proba, 4),
            "threshold": meta["threshold"], "prediction": int(proba >= meta["threshold"]),
            "notes": notes}


def run_predict(model_key=DEPLOYED_MODEL, json_path=None):
    if json_path:
        with open(json_path) as f:
            booking = {key: default for key, _, default, _ in QUESTIONS} | json.load(f)
    else:
        print("Enter the booking details (press Enter to keep the value in brackets).\n")
        booking = ask_user()
    result = predict_booking(booking, model_key)

    print("\n--- Prediction ---")
    print("Model:", result["model"])
    print(f"Probability of cancellation: {result['probability_of_cancel']:.1%}")
    print(f"Decision threshold: {result['threshold']:.2f}")
    print("Result:", "WILL LIKELY BE CANCELED (1)" if result["prediction"] else "will likely NOT be canceled (0)")
    for n in result["notes"]:
        print("Note:", n)
