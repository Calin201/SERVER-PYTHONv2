import os
import cv2
import numpy as np
import threading
import time
import argparse
import requests
import io
import queue
import json  # Added for JSON parsing
import random  # For simulating LiDAR data during development
from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS  # Import CORS
from personDetector import PersonDetector
from dotenv import load_dotenv
import socket
import subprocess
import re
import ipaddress
from pathlib import Path

# Global variables
# Default source URLs for video stream and sensors
source_url = 'http://192.168.1.205/stream'
boundary = '123456789000000000000987654321'
model_path = 'models/best.pt'

# Server configuration
# Default to 0.0.0.0 to listen on all network interfaces
host = '0.0.0.0'
# Or specify a custom IP address for the server
specific_ip = '192.168.72.254'
port = 5000

# Detector and stream state
detector = None
stream_thread = None
stream_active = False

# Implementare double buffering cu interpolare de frame-uri
class FrameBuffer:
    def __init__(self):
        self.buffer = None
        self.prev_buffer = None  # Frame anterior pentru interpolare
        self.lock = threading.Lock()
        self.last_update_time = 0
        
    def update(self, frame):
        with self.lock:
            # Salvăm frame-ul curent ca frame anterior înainte de actualizare
            if self.buffer is not None:
                self.prev_buffer = self.buffer
            self.buffer = frame
            self.last_update_time = time.time()
            
    def get(self):
        with self.lock:
            if self.buffer is not None:
                return self.buffer.copy()
            return None
    
    def get_interpolated(self, interpolation_factor=0.5):
        """Generează un frame interpolat între frame-ul curent și cel anterior
        
        Args:
            interpolation_factor: Factor de interpolare între 0 și 1 (0 = frame anterior, 1 = frame curent)
        
        Returns:
            Frame interpolat sau frame-ul curent dacă interpolarea nu este posibilă
        """
        with self.lock:
            if self.buffer is None:
                return None
                
            if self.prev_buffer is None:
                return self.buffer.copy()
                
            # Asigurăm că factorul este între 0 și 1
            factor = max(0, min(1, interpolation_factor))
            
            # Interpolare lineară între frame-uri
            try:
                # Verificăm dacă frame-urile au aceeași dimensiune
                if self.buffer.shape == self.prev_buffer.shape:
                    # Interpolare lineară între frame-uri
                    interpolated = cv2.addWeighted(self.prev_buffer, 1.0 - factor, self.buffer, factor, 0)
                    return interpolated
                else:
                    return self.buffer.copy()
            except Exception as e:
                print(f"Eroare la interpolarea frame-urilor: {e}")
                return self.buffer.copy()

# Buffere pentru frame-uri
original_buffer = FrameBuffer()
processed_buffer = FrameBuffer()

fps_display = 0.0  # Variabilă globală pentru FPS
last_error = None
connection_status = "Disconnected"
detections_data = {}  # Cache for the latest detection data

# Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def parse_mjpeg_stream(url, boundary):
    """
    Parse MJPEG stream with the specified boundary.
    
    Args:
        url (str): URL of the MJPEG stream
        boundary (str): Boundary string for multipart/x-mixed-replace
    
    Yields:
        bytes: JPEG frame data
    """
    global stream_active, connection_status, last_error
    
    try:
        # Make request to the stream URL
        connection_status = "Connecting..."
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code != 200:
            error_msg = f"Error: Received status code {response.status_code}"
            print(error_msg)
            connection_status = f"Error: HTTP {response.status_code}"
            last_error = error_msg
            return
        
        connection_status = "Connected"
        print(f"Connected to stream at {url}")
        
        # Set up buffer for reading the stream
        buffer = b''
        boundary_marker = f'--{boundary}'.encode()
        content_type_marker = b'Content-Type: image/jpeg'
        
        # Set a timeout for each chunk read
        chunk_timeout = 5  # seconds
        
        # Read the stream with timeout protection
        stream_iter = response.iter_content(chunk_size=1024)
        last_chunk_time = time.time()
        
        while stream_active:
            try:
                # Use next() with a timeout check instead of a for loop
                # This allows us to detect when the stream is stuck
                chunk = next(stream_iter, None)
                
                # Reset the chunk timeout since we received data
                last_chunk_time = time.time()
                
                # Check if we've reached the end of the stream
                if chunk is None:
                    print("End of stream reached")
                    break
                    
                buffer += chunk
                
                # Find boundary and content type markers
                start_idx = buffer.find(boundary_marker)
                if start_idx == -1:
                    # Keep the last part of the buffer in case it contains the start of a boundary
                    buffer = buffer[-len(boundary_marker):]
                    continue
                    
                content_idx = buffer.find(content_type_marker, start_idx)
                if content_idx == -1:
                    # Keep from the boundary marker onwards
                    buffer = buffer[start_idx:]
                    continue
                    
                # Find the start of the JPEG data
                jpeg_start = buffer.find(b'\r\n\r\n', content_idx) + 4
                if jpeg_start == 3:  # Not found
                    buffer = buffer[start_idx:]
                    continue
                    
                # Find the end of the JPEG data (next boundary marker)
                next_boundary_idx = buffer.find(boundary_marker, jpeg_start)
                if next_boundary_idx == -1:
                    # Not enough data yet
                    continue
                    
                # Extract the JPEG data
                jpeg_data = buffer[jpeg_start:next_boundary_idx]
                
                # Update buffer to contain only data from the next boundary onwards
                buffer = buffer[next_boundary_idx:]
                
                # Yield the JPEG frame
                yield jpeg_data
                
                # Check if we've been waiting too long for a chunk
                if time.time() - last_chunk_time > chunk_timeout:
                    print(f"No data received for {chunk_timeout} seconds, connection may be stalled")
                    break
                    
            except StopIteration:
                print("Stream iterator exhausted")
                break
                
            except requests.exceptions.ChunkedEncodingError as e:
                print(f"Chunked encoding error: {e}")
                break
                
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error while reading stream: {e}")
                break
                
    except requests.exceptions.RequestException as e:
        error_msg = f"Request error in stream parsing: {e}"
        print(error_msg)
        connection_status = "Disconnected"
        last_error = error_msg
        return
        
    except Exception as e:
        error_msg = f"Error in stream parsing: {e}"
        print(error_msg)
        connection_status = "Disconnected"
        last_error = error_msg
        return
        
    # If we get here, the stream has ended or errored out
    print("Stream parser exiting")
    connection_status = "Disconnected"

def process_stream():
    """
    Main function to process the video stream.
    """
    global original_frame, processed_frame, stream_active, detector, connection_status, last_error, detections_data, fps_display
    
    # Frame counter for processing every nth frame
    frame_id = 0
    last_process_time = time.time()
    process_interval = 0.1  # Process frames at 10 FPS max
    
    # For FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_update_interval = 1.0  # Update FPS every second
    
    # Add timeout detection for stream
    last_frame_time = time.time()
    max_frame_timeout = 10.0  # Maximum time to wait for a new frame before considering connection lost
    
    while stream_active:
        try:
            # Start a generator for parsing the stream
            stream_parser = parse_mjpeg_stream(source_url, boundary)
            
            # Process frames from the stream
            for jpeg_data in stream_parser:
                if not stream_active:
                    break
                    
                # Reset timeout since we received a frame
                last_frame_time = time.time()
                    
                # Decode JPEG data to OpenCV format
                frame_buffer = np.frombuffer(jpeg_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                
                # Update FPS calculation - doar la fiecare 10 frame-uri pentru a reduce overhead-ul
                if frame_id % 10 == 0:
                    fps_frame_count += 1
                    elapsed_time = time.time() - fps_start_time
                    if elapsed_time >= fps_update_interval:
                        fps_display = fps_frame_count * 10 / elapsed_time  # Înmulțim cu 10 pentru a compensa numărarea redusă
                        fps_frame_count = 0
                        fps_start_time = time.time()
                
                if frame is None:
                    print("Warning: Could not decode frame")
                    continue
                
                # Actualizare buffer original - fără blocare globală
                original_buffer.update(frame)
                
                # Only process frames at a certain interval to reduce CPU usage
                current_time = time.time()
                if current_time - last_process_time >= process_interval:
                    # Process frame for detection
                    if detector is not None and detector.frame_queue is not None:
                        # Only add to queue if not full to avoid backlog
                        if not detector.frame_queue.full():
                            # Folosim frame-ul original pentru detecție - evităm redimensionarea redundantă
                            # Detectorul va face propria redimensionare internă dacă este necesar
                            detector.frame_queue.put((frame, frame_id))
                            last_process_time = current_time
                
                # Try to get processed results if available
                if detector is not None and detector.results_queue is not None:
                    try:
                        if not detector.results_queue.empty():
                            results, orig_frame, frame_id_processed = detector.results_queue.get(block=False)
                            
                            # Get detections and visualization
                            # Obținem frame-ul original din buffer pentru procesare
                            orig_frame = original_buffer.get()
                            if orig_frame is not None:
                                # Calculate relative positions for API - fără blocare globală
                                _, detections = detector.calculate_relative_positions(orig_frame, results)
                                
                                # Debug print pentru a vedea dacă există detecții
                                if detections:
                                    print(f"Detected {len(detections)} objects: {[d['class'] for d in detections]}")
                                else:
                                    pass
                                    #print("No detections found")
                                    
                                # Store detections data for API endpoint
                                detections_data = {
                                    "frame_id": frame_id,
                                    "timestamp": time.time(),
                                    "fps": fps_display,
                                    "detections": detections
                                }
                                
                                # Draw debug visualization for UI with real FPS
                                vis_frame = detector.draw_debug_visualization(orig_frame, detections, fps_display)
                                # Actualizare buffer procesat
                                processed_buffer.update(vis_frame)
                    except queue.Empty:
                        pass
                
                frame_id += 1

                # Adaptive sleep to control CPU usage
                # Sleep longer if we're ahead of schedule
                elapsed = time.time() - current_time
                if elapsed < 0.05:  # Target ~20 FPS for UI updates
                    time.sleep(0.05 - elapsed)
                    
                # Check for timeout - break out of the loop if we've been waiting too long  
                # This will allow us to check if stream_active is still True and reconnect if needed
                if time.time() - last_frame_time > max_frame_timeout:
                    print(f"No frames received for {max_frame_timeout} seconds, reconnecting...")
                    connection_status = "Reconnecting..."
                    last_error = "Connection timeout - no frames received"
                    break
            
            # If we exit the loop and stream is still active, try to reconnect after a delay
            if stream_active:
                print("Stream ended or timed out, attempting to reconnect in 5 seconds...")
                connection_status = "Reconnecting..."
                time.sleep(5)
            
        except Exception as e:
            error_msg = f"Error in stream processing: {e}"
            print(error_msg)
            connection_status = "Error"
            last_error = error_msg
            
            # Wait before retrying
            time.sleep(5)

def generate_mjpeg(frame_source):

    global fps_display

    while True:

        if frame_source == 'original':
            frame = original_buffer.get()
            
            if frame is not None:
                h, w = frame.shape[:2]
                fps_text = f'FPS: {fps_display:.1f}'
                text_position = (w - 150, 40)
                cv2.putText(frame, fps_text, text_position, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        elif frame_source == 'processed':
            frame = processed_buffer.get()

        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        


@app.route('/')
def index():
    """Render the main page with stream viewers and controls."""
    return render_template('petoi_control.html', source_url=source_url)

@app.route('/stream/original')
def stream_original():
    return Response(generate_mjpeg('original'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream/processed')
def stream_processed():
    return Response(generate_mjpeg('processed'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/object_positions')
def object_positions():
    """API endpoint to get the current object positions."""
    global detections_data
    
    # Verificăm dacă avem date
    if not detections_data:
        return jsonify({"detections": []})
    
    # Pregătim datele pentru frontend
    current_time = time.time()
    # Using datetime module for proper microsecond formatting
    from datetime import datetime
    timestamp = datetime.fromtimestamp(current_time).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
    
    # Creăm o copie îmbogățită a datelor
    enhanced_data = {
        "id": int(current_time * 100) % 1000,  # ID unic generat din timestamp
        "timestamp": timestamp,
        "frame_id": detections_data.get("frame_id", 0),
        "fps": detections_data.get("fps", 0),
        "unix_timestamp": int(current_time),
        "detections": []
    }
    
    # Procesare detecții - convertim clasele pentru compatibilitate
    for detection in detections_data.get("detections", []):
        # Verificăm dacă detectarea are toate datele necesare
        if "class" in detection and "confidence" in detection:
            # Copiem detectarea pentru a nu modifica originalul
            processed_detection = detection.copy()
            
            # Putem mapa anumite clase pentru compatibilitate cu frontend-ul
            # De exemplu, frontend-ul ar putea să se aștepte doar la clasa "person"
            # class_mapping = {"block25": "person", "other_class": "person"}
            # processed_detection["class"] = class_mapping.get(detection["class"], detection["class"])
            
            # Adăugăm detectarea procesată
            enhanced_data["detections"].append(processed_detection)
    
    # Return the enhanced detection data
    return jsonify(enhanced_data)

@app.route('/api/source', methods=['POST'])
def update_source():
    """API endpoint to update the source URL."""
    global source_url, stream_active, stream_thread
    
    data = request.json
    new_url = data.get('url')
    
    if not new_url:
        return jsonify({"error": "No URL provided"}), 400
    
    # Update source URL
    source_url = new_url
    
    # Restart stream thread
    restart_stream()
    
    return jsonify({"status": "success", "url": source_url})

def restart_stream():
    """Restart the stream processing thread."""
    global stream_active, stream_thread
    
    # Stop current stream if active
    stream_active = False
    if stream_thread and stream_thread.is_alive():
        stream_thread.join(timeout=1.0)
    
    # Start new stream thread
    stream_active = True
    stream_thread = threading.Thread(target=process_stream, daemon=True)
    stream_thread.start()

def start_detector():
    """Initialize the detector and start its inference thread."""
    global detector
    
    try:
        # Initialize detector
        detector = PersonDetector(
            model_path=model_path,
            device='cuda',
            imgsz=640
        )
        
        # Start inference thread
        inference_thread = threading.Thread(target=detector.inference_thread)
        inference_thread.daemon = True
        inference_thread.start()
        
        print("Object Position Detector initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return False

def init_app():
    """Initialize the application."""
    global detector, stream_active, stream_thread, discovery_active, discovery_thread
    
    # Initialize the YOLOv8 detector
    start_detector()
    
    # Start stream thread
    stream_active = True
    stream_thread = threading.Thread(target=process_stream, daemon=True)
    stream_thread.start()
    
    return True


@app.route('/bt-control', methods=['GET', 'POST'])
def bt_control():
    # Support both GET query parameters and POST JSON/body
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        command = data.get('command') or data.get('cmd', '')
        orientation = data.get('orientation', '')
    else:
        command = request.args.get('cmd', '')
        orientation = request.args.get('orientation', '')

    print(f"Received BT Command: {command}, Orientation: {orientation}")
    try:
        robot_url = source_url.rsplit('/', 1)[0] + '/bt-control?cmd=' + command + '&orientation=' + orientation
        requests.get(robot_url, timeout=2.0)
        return jsonify(status='success', command=command, orientation=orientation)
    except Exception as e:
        print(f"Error sending command to robot: {e}")
        return jsonify(status='error', command=command, orientation=orientation), 500


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MJPEG Stream Processor with Adaptive Detection')
    parser.add_argument('--source', type=str, default=source_url,
                       help='Source MJPEG stream URL')
    parser.add_argument('--model', type=str, default=model_path,
                       help='Path to YOLOv8 model')

    

    args = parser.parse_args()

    # Update global variables from arguments
    source_url = args.source
    model_path = args.model

    available_ips = []
    available_ips.append('0.0.0.0')  # Always allow binding to all interfaces
    available_ips.append('127.0.0.1')  # Always allow localhost
    

    host = '0.0.0.0'
    port = 5000
    

    # Initialize app
    if init_app():
        # Determine the actual URL to access the application
        access_url = f"http://{host}:{port}"
        if host == '0.0.0.0':
            # If binding to all interfaces, suggest using the machine's actual IP
            print(f"\nApplication will be accessible at:")
            print(f"  http://{host}:{port}")

        # Run Flask app
        print(f"\nStarting web server at {access_url}")
        try:
            app.run(host=host, port=port, threaded=True)
        except OSError as e:
            if "The requested address is not valid in its context" in str(e):
                print(f"\nError: Cannot bind to IP address '{host}' because it doesn't exist on this system.")
                print("Available IP addresses:")
                for adapter, ip in available_interfaces:
                    print(f"  {adapter}: {ip}")
                print("\nTry running the app with one of these IP addresses, for example:")
                if available_interfaces:
                    suggested_ip = available_interfaces[0][1]
                    print(f"  python app.py --specific-ip {suggested_ip}")
                print("Or use 0.0.0.0 to listen on all interfaces:")
                print("  python app.py --host 0.0.0.0")
            else:
                print(f"\nError starting web server: {e}")