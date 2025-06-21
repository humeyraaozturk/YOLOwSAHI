from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from sahi.prediction import PredictionResult

model_path = "model/best.pt"
image_path = "your/test/img.jpg"
output_path = "output"


detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path= model_path,
    confidence_threshold=0.3,
    device="cuda:0"    
)

result = get_prediction(image_path, detection_model)
result.export_visuals(export_dir = output_path)
object_prediction_list = result.object_prediction_list

import json
all_detections = []

for i, pred in enumerate(object_prediction_list):
    all_detections.append({
        "index": i,
        "category name": pred.category.name,
        "category id": pred.category.id,
        "score": float(pred.score.value),
        "bbox": list(map(float, pred.bbox.to_xywh()))  # [x, y, w, h]
    })
    
with open(f"{output_path}\\detections.json", "w") as f:
    json.dump(all_detections, f, indent=4)
    
    
# with open(output_path + "detections.jsonl", "w") as f:
#     for detection in all_detections:
#         json.dump(detection, f)
#         f.write('\n')