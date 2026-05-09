# Step 4 - Detection Filtering and Decision Logic

## Goal
Introduce filtering and decision-making on top of YOLO detections.

## Actions
Selected relevant classes (person, car, bus, truck).
Implemented filtering logic to keep only useful detections.
Added a decision layer to count objects and trigger alerts.

## Result
The system now performs basic scene understanding:
- detects people and vehicles
- counts them
- generates alerts such as crowd detection and heavy traffic

## Next Step
Connect detection results to a communication layer (MQTT) for multi-agent interaction.