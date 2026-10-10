from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data="kitti.yaml",
        epochs=50,
        imgsz=640,
        project="runs/detect",
        name="yolov8_kitti"
    )

if __name__ == "__main__":
    main()