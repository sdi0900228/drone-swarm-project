# Step 3 - Multiple Sample Image Tests

## Goal
Extend the baseline YOLO test so that it can process multiple sample images instead of only one.

## Actions
Added more sample images in the data/samples folder.
Updated the baseline detection script to iterate through all supported image files in the sample folder.
Ran the script again and tested object detection on multiple images.

## Result
The detection workflow now supports repeated baseline testing on several images, making the setup more realistic and useful for future dataset preparation.

## Next Step
Review the detection outputs, keep the most useful sample cases, and prepare the transition to dataset collection and labeling.