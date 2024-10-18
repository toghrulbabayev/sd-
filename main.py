import argparse
import torch
import cv2
import os

def detect(source='data/videos', weights='yolov5s.pt', save_results=True):
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Open video capture
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {source}")
        return

    # Create a directory for saving results if needed
    if save_results:
        os.makedirs('runs/detect', exist_ok=True)
        output_file = 'runs/detect/output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
        out = cv2.VideoWriter(output_file, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))  # 30 FPS

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Break if no frames are returned

        # Perform inference
        results = model(frame)

        # Print results
        results.print()  # Print results to the console
        annotated_frame = results.render()[0]  # Render results on the frame

        # Show the frame with detections
        cv2.imshow('YOLOv5 Detection', annotated_frame)

        # Write the frame to the output video if save_results is True
        if save_results:
            out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if save_results:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description='YOLOv5 Object Detection from Video')
    parser.add_argument('--source', type=str, default='data/videos', help='Path to video file (default: data/videos)')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Path to weights file (default: yolov5s.pt)')
    parser.add_argument('--save', action='store_true', help='Save results to output video file')

    args = parser.parse_args()

    # Run detection
    detect(source=args.source, weights=args.weights, save_results=args.save)
