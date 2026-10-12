from ultralytics import RTDETR

def main():
    model = RTDETR("rtdetr-l.pt")
    
    results = model.train(
        data="kitti.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        patience=10,
        project="runs/detect",
        name="rtdetr_kitti"
    )

if __name__ == "__main__":
    main()