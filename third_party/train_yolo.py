# train_yolo.py
from ultralytics import YOLO
import multiprocessing as mp

def main():
    model = YOLO('yolov8s.pt')
    model.train(
        data='data.yaml',
        imgsz=640,
        epochs=60,
        batch=-1,
        device=0,
        workers=2,   # you can keep >0 now
        cache='disk', # or 'ram'
        name='traffic_v8s_fhwalabels'
    )

if __name__ == '__main__':
    mp.freeze_support()         # needed on Windows when spawning
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
