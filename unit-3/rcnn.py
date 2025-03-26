import cv2
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

# Load pre-trained backbone
backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone("resnet50", pretrained=True)

# Define Faster R-CNN model
model = FasterRCNN(backbone, num_classes=2)  # 2 classes: object + background
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
CLASS_LABELS = {0: "Background", 1: "Object"}
def transform_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = F.to_tensor(image)
    return image

def detect_faster_rcnn(frame):
    image = transform_image(frame)
    with torch.no_grad():
        prediction = model([image])
    
    for box, score, label in zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels']):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box.tolist())
            class_name = CLASS_LABELS.get(label.item(), "Unknown")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Capture video
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    rcnn_frame = detect_faster_rcnn(frame.copy())
    
    cv2.imshow("Faster R-CNN Detection", rcnn_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()