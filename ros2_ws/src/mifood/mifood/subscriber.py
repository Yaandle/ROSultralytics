from ultralytics import YOLO
import cv2
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        self.publisher_ = self.create_publisher(String, 'detection_results', 10)
        self.timer = self.create_timer(0.5, self.process_video)
        package_dir = os.path.dirname(os.path.dirname(__file__))
        self.model_path = os.path.join(package_dir, 'Models', 'Strawberrymodel.pt')
        self.video_path = os.path.join(package_dir, 'resources', 'strawberryvideo.avi')
        
        self.model = YOLO(self.model_path)
        self.cap = cv2.VideoCapture(self.video_path)
    
    def process_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            cv2.destroyAllWindows()
            self.destroy_node()
            return

        results = self.model(frame)
        for *xyxy, conf, cls in results.xyxy[0]:
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            bbox_data = {
                'x1': int(xyxy[0]),
                'y1': int(xyxy[1]),
                'x2': int(xyxy[2]),
                'y2': int(xyxy[3]),
                'confidence': float(conf),
                'class_id': int(cls)
            }
            self.publisher_.publish(String(data=json.dumps(bbox_data)))

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisher()
    rclpy.spin(video_publisher)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
