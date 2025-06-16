import cv2
import torch
import numpy as np
from ultralytics import YOLO
import argparse
import time
import numpy as np
import os
from threading import Thread
import queue
import json

class PersonDetector:
    def __init__(self, model_path, device='cuda', imgsz=640):
        """
        Initialize the PersonDetector.
        
        Args:
            model_path: Path to the YOLO model
            device: Device to run inference on ('cuda' or 'cpu')
            imgsz: Image size for inference
        """
        self.model = YOLO(model_path)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.imgsz = imgsz
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Class colors for visualization (only used for debug)
        self.class_colors = {
            # Actualizare pentru clasele specifice modelului
            "person": (0, 0, 255),       # Red (BGR format)
        }
        
        # Queues for threaded processing
        self.frame_queue = queue.Queue(maxsize=4)
        self.results_queue = queue.Queue(maxsize=4)
        
        # Warmup GPU - opțional, poate fi dezactivat pentru a evita probleme de memorie
        try:
            if self.device == 'cuda':
                print("Warming up GPU...")
                dummy_input = torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device) # folosim zeros în loc de randn pentru a consuma mai puțină memorie
                _ = self.model(dummy_input) # Doar o singură încălzire în loc de 3
                print("GPU warmup complete")
        except Exception as e:
            print(f"Warning: GPU warmup failed, but continuing without it: {e}")
            # Continuăm fără încălzire

    def inference_thread(self):
        """Thread for running inference"""
        print("Inference thread started")
        print(f"Model info: {self.model}")
        print(f"Classes that can be detected: {self.model.names}")
        print(f"Object Position Detector initialized successfully")
        
        while True:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.001)  # Small sleep to prevent CPU hogging
                    continue
                    
                frame_data = self.frame_queue.get()
                if frame_data is None:  # Signal to stop the thread
                    break
                    
                frame, frame_id = frame_data
                
                # Salvăm dimensiunile originale pentru a putea scala corect detecțiile înapoi
                orig_shape = frame.shape
                
                # Run inference pe frame cu imgsz specificat (de obicei 640x640)
                # Importante: dimensiunile de intrare sunt probabil diferite de dimensiunile de ieșire
                # și asta creează problema de offset
                results = self.model(frame, imgsz=self.imgsz, verbose=False)[0]
                
                # Pentru diagnoză, salvăm informații relevante în rezultate pentru a le folosi mai târziu
                # la scalarea corectă a coordonatelor
                if not hasattr(results, 'orig_shape'):
                    results.orig_shape = orig_shape
                if not hasattr(results, 'input_shape'):
                    if isinstance(self.imgsz, int):
                        results.input_shape = (self.imgsz, self.imgsz)
                    else:
                        results.input_shape = self.imgsz
                
                # Verifică dacă avem detecții și afișează informații despre ele
                if hasattr(results, 'boxes') and len(results.boxes) > 0:
                    for i, box in enumerate(results.boxes):
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        name = self.model.names[cls]
                else:
                    pass
                
                # Put results in queue
                self.results_queue.put((results, frame, frame_id))
            except Exception as e:
                print(f"Error in inference thread: {e}")
                import traceback
                traceback.print_exc()
        print("Inference thread stopped")
    
    def calculate_relative_positions(self, frame, results):
        """
        Calculate the relative positions of detected objects.
        
        Args:
            frame: Input frame
            results: Detection results from the model
            
        Returns:
            A tuple containing:
            - The original frame
            - A list of detection data with relative positions
        """
        detections = []
        
        # Dimensiunea reală a frame-ului
        h, w = frame.shape[:2]
        
        # Precizie mai mare pentru debug
        #print(f"\nDebug - Frame dimensions: {w}x{h}")
        
        # Determinăm dimensiunile utilizate pentru modelul de inferență
        if isinstance(self.imgsz, int):
            model_h = model_w = self.imgsz
        elif isinstance(self.imgsz, (list, tuple)) and len(self.imgsz) == 2:
            model_w, model_h = self.imgsz
        else:
            # Default la 640x640 dacă nu sunt siguri
            model_w = model_h = 640
            
        # Fix special pentru imaginile de la YOLO care au aspect ratio diferit
        # YOLO redimensionează imaginea păstrând aspect ratio-ul, adăugând padding
        img_ratio = w / h
        model_ratio = model_w / model_h
        
        # Ajustăm factorii de scalare ținând cont de padding-ul posibil
        if img_ratio > model_ratio:  # imaginea este mai lată decât modelul
            scale_y = h / (model_w / img_ratio)  # height ajustat pentru padding
            scale_x = w / model_w
        else:  # imaginea este mai înaltă decât modelul
            scale_x = w / (model_h * img_ratio)  # width ajustat pentru padding
            scale_y = h / model_h
        
        #print(f"Debug - Model dimensions: {model_w}x{model_h}")
        #print(f"Debug - Scale factors: X={scale_x:.4f}, Y={scale_y:.4f}")
        
        # Offset-uri pentru ajustare fină - calibrate pentru alinierea corectă cu obiectele
        # Valori negative deplasează boxurile spre stânga/sus, valori pozitive spre dreapta/jos
        offset_x = 30    # A fost -20, acum 0 pentru a nu mai deplasa spre stânga
        offset_y = 30   # Deplasare ușoară spre sus
        
        if hasattr(results, 'boxes'):
            boxes = results.boxes
            num_boxes = len(boxes)
            #print(f"Debug - Number of detections: {num_boxes}")
            
            for i, box in enumerate(boxes):
                # Extract coordinates, confidence, and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Get class name
                class_name = self.model.names[cls]
                
                # Scalarea precisă pentru conversia coordonatelor de la modelul de inferenta la frame-ul real
                # Adăugăm offset-uri pentru calibrare fină
                adjusted_x1 = x1 * scale_x + offset_x
                adjusted_y1 = y1 * scale_y + offset_y
                adjusted_x2 = x2 * scale_x + offset_x
                adjusted_y2 = y2 * scale_y + offset_y
                
                if i == 0:  # Print info pentru primul obiect detectat (debug)
                    pass
                    #print(f"Debug - Detection {i+1} ({class_name})")
                    #print(f"  Original coords: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
                    #print(f"  Adjusted coords: ({adjusted_x1:.1f}, {adjusted_y1:.1f}) - ({adjusted_x2:.1f}, {adjusted_y2:.1f})")
                
                # Calculate center position (adjusted)
                center_x = (adjusted_x1 + adjusted_x2) / 2
                center_y = (adjusted_y1 + adjusted_y2) / 2
                
                # Calculate relative positions (0-1 scale)
                rel_x = center_x / w
                rel_y = center_y / h
                
                # Calculate bounding box width and height (adjusted)
                width = adjusted_x2 - adjusted_x1
                height = adjusted_y2 - adjusted_y1
                
                # Calculate relative width and height
                rel_width = width / w
                rel_height = height / h
                
                # Add detection to the list with adjusted coordinates
                detection = {
                    "class": class_name,
                    "confidence": conf,
                    "position": {
                        "center": {
                            "x": float(rel_x),
                            "y": float(rel_y)
                        },
                        "width": float(rel_width),
                        "height": float(rel_height)
                    },
                    "absolute": {
                        "center": {
                            "x": float(center_x),
                            "y": float(center_y)
                        },
                        "x1": float(adjusted_x1),
                        "y1": float(adjusted_y1),
                        "x2": float(adjusted_x2),
                        "y2": float(adjusted_y2)
                    }
                }
                detections.append(detection)
        
        return frame, detections

    def process_stream(self, source=0, show_debug=False, save=False, output_path='output.mp4'):
        """
        Process video stream with real-time detection using threading.
        
        Args:
            source: Video source (0 for webcam, URL/IP for stream, or video file path)
            show_debug: Show debug visualization of detections
            save: Save output to file
            output_path: Path to save output video if save is True
        """
        # Handle different source types
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            print(f"Opening webcam {source}...")
        else:
            print(f"Opening video source: {source}")
            
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        # Get stream properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:  # For webcams, fps might be 0
            fps = 30
            
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {w}x{h}, {fps} fps")
        
        # Initialize VideoWriter if saving
        if save:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            print(f"Saving output to {output_path}")
        
        # Start inference thread
        inference_thread = Thread(target=self.inference_thread, daemon=True)
        inference_thread.start()
        
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        fps_update_time = start_time
        fps_display = 0
        last_frame_id = -1
        
        print("Starting detection...")
        print("Press 'q' to quit, 's' to save a screenshot")
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("End of stream or error reading frame")
                    break
                
                # Put frame in queue for inference
                if not self.frame_queue.full():
                    self.frame_queue.put((frame, frame_count))
                
                # Get results if available
                try:
                    if not self.results_queue.empty():
                        results, orig_frame, frame_id = self.results_queue.get(block=False)
                        
                        # Calculate relative positions
                        _, detections = self.calculate_relative_positions(orig_frame, results)
                        
                        # Update FPS calculation every second
                        processed_count += 1
                        current_time = time.time()
                        if current_time - fps_update_time >= 1.0:
                            fps_display = processed_count / (current_time - start_time)
                            fps_update_time = current_time
                        
                        # Create response data
                        response_data = {
                            "frame_id": frame_id,
                            "timestamp": time.time(),
                            "fps": fps_display,
                            "detections": detections
                        }
                        
                        # Print the response data as JSON
                        json_response = json.dumps(response_data, indent=2)
                        print(json_response)
                        
                        # Show debug visualization if requested
                        if show_debug:
                            debug_frame = self.draw_debug_visualization(orig_frame, detections, fps_display)
                            cv2.imshow('Object Position Detector', debug_frame)
                            
                        # Save frame if requested
                        if save:
                            # If showing debug visualization, save that, otherwise save original
                            if show_debug:
                                out.write(debug_frame)
                            else:
                                out.write(orig_frame)
                            
                        last_frame_id = frame_id
                except queue.Empty:
                    pass
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s') and show_debug:
                    # Save screenshot
                    screenshot_dir = "screenshots"
                    os.makedirs(screenshot_dir, exist_ok=True)
                    screenshot_path = os.path.join(screenshot_dir, f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(screenshot_path, debug_frame)
                    print(f"\nScreenshot saved to {screenshot_path}")
                
                frame_count += 1
                
                # Add a small sleep to prevent CPU hogging
                time.sleep(0.001)
        
        finally:
            # Signal inference thread to stop
            self.frame_queue.put(None)
            inference_thread.join(timeout=1.0)
            
            # Clean up
            cap.release()
            if save:
                out.release()
            cv2.destroyAllWindows()
            
            # Print summary
            elapsed_time = time.time() - start_time
            print(f"\nProcessed {processed_count} frames in {elapsed_time:.2f} seconds ({processed_count/elapsed_time:.1f} fps)")

    def draw_debug_visualization(self, frame, detections, fps):
        """
        Draw debug visualization of detections on the frame.
        Only used when show_debug is True.
        
        Args:
            frame: Input frame
            detections: List of detection data
            fps: Current FPS value
            
        Returns:
            Frame with debug visualization
        """
        debug_frame = frame.copy()
        h, w = debug_frame.shape[:2]
        
        # Add FPS counter în colțul din dreapta sus
        cv2.putText(
            debug_frame, 
            f'FPS: {fps:.1f}', 
            (w - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,
            (0, 255, 0),
            1
        )
        
        # Procesez detecțiile
        # Determină factorul de scalare între dimensiunea reală a frame-ului și dimensiunea folosită pentru inferență
        # Acest factor ajută la alinierea corectă a bounding box-urilor
        scale_x = 1.0
        scale_y = 1.0
        
        model_size = self.imgsz
        if isinstance(model_size, int):
            # Dacă model_size este un singur număr (pătrățel), calculăm raportul pentru fiecare dimensiune
            scale_x = w / model_size
            scale_y = h / model_size
        
        for detection in detections:
            class_name = detection["class"]
            conf = detection["confidence"]
            
            # Extragem valorile absolute (acestea sunt în coordonate originale)
            abs_pos = detection["absolute"]
            
            # Coordonatele originale trebuie ajustate în funcție de raportul dintre dimensiunea frame-ului și cea a modelului
            # Aplicăm offset-ul și scalarea pentru a alinia corect box-urile
            x1 = abs_pos["x1"]
            y1 = abs_pos["y1"]
            x2 = abs_pos["x2"]
            y2 = abs_pos["y2"]
            center_x = abs_pos["center"]["x"]
            center_y = abs_pos["center"]["y"]
            
            # Get color for this class
            color = self.class_colors.get(class_name, (128, 128, 128))
            
            # Desenez conturul cu linie subțire (1px)
            cv2.rectangle(
                debug_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                1  # Linie subțire
            )
            
            # Desenez punctul central
            cv2.circle(
                debug_frame,
                (int(center_x), int(center_y)),
                4,  # Radius mai mic
                color,
                -1
            )
            
            # Adaug eticheta text simplificată
            label = f'{class_name} {conf:.2f}'
            
            # Fundal pentru text minimal
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                debug_frame, 
                (int(x1), int(y1) - 15), 
                (int(x1) + text_w, int(y1)), 
                color, 
                -1
            )
            
            # Text mai mic și mai simplu
            cv2.putText(
                debug_frame,
                label,
                (int(x1), int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font mai mic
                (255, 255, 255),
                1  # Grosime mai mică
            )
        
        return debug_frame

    def process_single_frame(self, frame):
        """
        Process a single frame and return the detections.
        Useful for API endpoints.
        
        Args:
            frame: Input frame
            
        Returns:
            A tuple containing:
            - The original frame
            - A list of detection data with relative positions
        """
        try:
            # Run inference on the original frame without any preprocessing
            results = self.model(frame, imgsz=self.imgsz, verbose=False)[0]
            
            # Calculate and return relative positions
            return self.calculate_relative_positions(frame, results)
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, []


def main():
    parser = argparse.ArgumentParser(description='Object position detector')
    parser.add_argument('--model', type=str, default='best.pt',
                        help='Path to trained model (best.pt)')
    parser.add_argument('--source', type=str, default='86.121.67.198',
                        help='Video source (0 for webcam, URL/IP for stream, or video file path)')
    parser.add_argument('--conf', type=float, default=0.05,
                        help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.35,
                        help='IoU threshold (0-1)')
    parser.add_argument('--save', action='store_true',
                        help='Save results to file')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug visualization')
    parser.add_argument('--output', type=str, default='output_videos/object_position.mp4',
                        help='Output video path (if save is enabled)')
    parser.add_argument('--imgsz', type=int, default=480,
                        help='Inference size (pixels)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    
    args = parser.parse_args()
    
    detector = PersonDetector(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device=args.device,
        imgsz=args.imgsz
    )
    
    detector.process_stream(
        source=args.source,
        show_debug=args.debug,
        save=args.save,
        output_path=args.output
    )


if __name__ == "__main__":
    main()