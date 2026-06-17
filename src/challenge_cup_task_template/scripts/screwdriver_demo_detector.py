import cv2
import numpy as np
from ultralytics import YOLO


class ScrewdriverDemoDetector:
    def __init__(self, model_path, conf=0.25, imgsz=640, device=0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect_screwdriver(self, bgr_image):
        results = self.model.predict(
            source=bgr_image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )

        result = results[0]
        detections = []

        if result.boxes is None or result.masks is None:
            return detections

        names = result.names
        boxes = result.boxes
        masks = result.masks.data.cpu().numpy()

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = names[cls_id]
            score = float(boxes.conf[i].item())

            if cls_name != "screwdriver":
                continue

            mask = (masks[i] > 0.5).astype(np.uint8)
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()

            center = self.get_mask_center(mask)

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": score,
                "bbox_xyxy": xyxy,
                "mask": mask,
                "center_uv": center,
            })

        return detections

    @staticmethod
    def get_mask_center(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        u = float(np.mean(xs))
        v = float(np.mean(ys))
        return u, v


if __name__ == "__main__":
    model_path = "models/yolo/yolov8n_seg_screwdriver_demo.pt"
    image_path = "test_scene2.png"

    detector = ScrewdriverDemoDetector(
        model_path=model_path,
        conf=0.25,
        imgsz=640,
        device=0,
    )

    img = cv2.imread(image_path)
    dets = detector.detect_screwdriver(img)

    print("detections:", len(dets))

    for d in dets:
        print(d["class_name"], d["confidence"], d["center_uv"], d["bbox_xyxy"])

        mask = d["mask"]
        img[mask > 0] = img[mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5

        if d["center_uv"] is not None:
            u, v = d["center_uv"]
            cv2.circle(img, (int(u), int(v)), 5, (0, 255, 0), -1)

    cv2.imwrite("screwdriver_demo_result.png", img)
    print("saved to screwdriver_demo_result.png")
