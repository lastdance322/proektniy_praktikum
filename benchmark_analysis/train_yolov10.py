from ultralytics import YOLO

def main():
    model = YOLO("yolov10n.pt") 
    
    results = model.train(
        data="kitti.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        patience=10,
        project="runs/detect",
        name="yolov10_kitti"
    )

if __name__ == "__main__":
    main()