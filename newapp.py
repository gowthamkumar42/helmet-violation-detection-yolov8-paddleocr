import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
import pandas as pd
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# --- Setup GPU for TensorFlow ---
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# --- Load YOLOv3 (with safe backend fallback) ---
yolo_weights = "yolov3-custom_7000.weights"
yolo_cfg = "yolov3-custom.cfg"

net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# Try to set CUDA backend/target if OpenCV supports it; otherwise fall back to CPU
use_cuda = False
try:
    # Try to enable CUDA — if OpenCV lacks CUDA DNN support this may raise/assert
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # Validate by asking the set values (no guarantees but often safe)
    # If the OpenCV build doesn't support it, the call above will usually raise.
    use_cuda = True
    print("YOLO: requested CUDA backend/target.")
except Exception as e:
    # Fallback to CPU
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception:
        # Best-effort fallback (some old OpenCV versions ignore these)
        pass
    print("YOLO: CUDA backend not available or failed to set. Falling back to CPU. (Reason: {})".format(e))

# --- Load helmet CNN (safe load and rebuild input if needed) ---
cnn_path = 'helmet-nonhelmet_cnn.h5'
try:
    base_model = load_model(cnn_path, compile=False)
    # Try to wrap with a fixed Input shape - commonly resolves input-rank mismatch when saved config lost input shape
    try:
        inp = Input(shape=(224, 224, 3))
        out = base_model(inp)
        model = Model(inputs=inp, outputs=out)
        print("Helmet CNN loaded and wrapped with Input(224,224,3).")
    except Exception as e:
        # if wrapping fails, fallback to the loaded model (maybe it already works)
        model = base_model
        print("Helmet CNN loaded but couldn't wrap with Input. Using raw loaded model. (Reason: {})".format(e))
except Exception as e:
    print("Failed to load helmet CNN model '{}'. Error: {}".format(cnn_path, e))
    raise

# --- Load TrOCR for OCR ---
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trocr_model.to(device)
print("TrOCR model loaded and moved to device:", device)

# --- Video setup ---
cap = cv2.VideoCapture('22.mp4')
if not cap.isOpened():
    raise RuntimeError("Could not open video file '22.mp4'")

# Read first frame to get sizes and to initialize writer properly
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Couldn't read first frame from video.")

first_frame = imutils.resize(first_frame, height=500)
frame_height, frame_width = first_frame.shape[:2]

# create writer with actual frame size
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter('output.avi', fourcc, 5, (frame_width, frame_height))

# COLORS for classes (assumes classId 0=bike, else plate)
COLORS = [(0, 255, 0), (0, 0, 255)]

# Prepare YOLO output layer names safely
layer_names = net.getLayerNames()
try:
    outs_indices = net.getUnconnectedOutLayers()
    # getUnconnectedOutLayers can return shape (N,1) or (N,) — normalize
    outs_flat = [int(i) for i in np.array(outs_indices).reshape(-1)]
    output_layers = [layer_names[i - 1] for i in outs_flat]
except Exception as e:
    # fallback: use all layers (slower/less correct but safe)
    output_layers = layer_names
    print("Warning: couldn't get YOLO unconnected out layers; falling back to all layer names. Reason:", e)

# --- Store no-helmet data ---
no_helmet_data = []

# utility: safe imshow (handles headless opencv)
_can_show = True
try:
    cv2.namedWindow("test_window")
    cv2.imshow("test_window", np.zeros((2,2,3), dtype=np.uint8))
    cv2.waitKey(1)
    cv2.destroyWindow("test_window")
except Exception:
    _can_show = False
    print("OpenCV GUI not available in this environment — frames will be saved instead of shown.")

# --- Helmet prediction function ---
def helmet_or_nohelmet(helmet_roi):
    try:
        # Skip empty ROIs
        if helmet_roi is None or helmet_roi.size == 0:
            return 0

        # Resize and ensure 3 channels
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        if helmet_roi.ndim == 2:  # grayscale
            helmet_roi = cv2.cvtColor(helmet_roi, cv2.COLOR_GRAY2RGB)
        elif helmet_roi.shape[-1] == 4:  # RGBA/BGRA
            helmet_roi = cv2.cvtColor(helmet_roi, cv2.COLOR_BGRA2BGR)

        # Normalize and expand batch dimension
        helmet_roi = helmet_roi.astype('float32') / 255.0
        helmet_roi = np.expand_dims(helmet_roi, axis=0)  # (1, 224, 224, 3)

        # Predict
        pred = model.predict(helmet_roi, verbose=0)
        # pred shape depends on your model; assume single sigmoid neuron or single probability
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        # handle shape like (1,1) or (1,) or (1,2) etc.
        val = None
        if pred.ndim == 2 and pred.shape[1] == 1:
            val = float(pred[0][0])
        elif pred.ndim == 1:
            val = float(pred[0])
        elif pred.ndim == 2 and pred.shape[1] == 2:
            # maybe softmax of two classes => take argmax
            val = float(np.argmax(pred[0]))  # 0 or 1
            # convert to a 0/1 as expected below
            return int(val)
        else:
            # fallback - take first element
            val = float(np.ravel(pred)[0])

        # Interpret: if model outputs probability (0..1), round: 0=helmet, 1=no-helmet
        return int(round(val))
    except Exception as e:
        print("Helmet CNN error:", e)
        return 0

# --- OCR function ---
def extract_plate_text(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = trocr_model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    except Exception as e:
        print("OCR error:", e)
        return "ERROR"

# reset video to first frame (we already read it)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# --- Process video frames ---
frame_idx = 0
while True:
    ret, img = cap.read()
    if not ret:
        break
    frame_idx += 1

    img = imutils.resize(img, height=500)
    height, width = img.shape[:2]

    # YOLO blob
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # run forward (wrapped in try so we can capture backend issues)
    try:
        outs = net.forward(output_layers)
    except Exception as e:
        # If CUDA backend caused a runtime error, try switching to CPU on the fly
        print("YOLO forward error:", e)
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            outs = net.forward(output_layers)
            print("Switched YOLO to CPU backend and continued.")
        except Exception as e2:
            print("Failed to run YOLO forward even on CPU. Error:", e2)
            break

    confidences, boxes, classIds = [], [], []

    # outs may be list of arrays
    for out in outs:
        for detection in out:
            scores = detection[5:]
            if scores.size == 0:
                continue
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = max(0, int(center_x - w / 2))
                y = max(0, int(center_y - h / 2))
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(class_id)

    indexes = []
    if len(boxes) > 0:
        try:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            # Normalize indexes (could be list of ints or numpy array)
            if isinstance(indexes, (np.ndarray, list)):
                indexes = np.array(indexes).reshape(-1).tolist()
            else:
                # e.g., in some OpenCV versions it returns tuples; try to iterate safely
                indexes = [int(i) for i in indexes]
        except Exception:
            # If NMSBoxes signature differs, fallback to taking all boxes
            indexes = list(range(len(boxes)))

    # For this frame: track whether any bike was detected without helmet
    frame_no_helmet_found = False

    for i in range(len(boxes)):
        if i not in indexes:
            continue

        x, y, w, h = boxes[i]
        color = [int(c) for c in COLORS[classIds[i] % len(COLORS)]]

        if classIds[i] == 0:  # bike
            # crop region near top quarter where rider/helmet likely appears
            try:
                helmet_roi = img[y: y + max(1, h // 4), x: x + max(1, w)]
            except Exception:
                helmet_roi = img[y:y+h, x:x+w]
            c = helmet_or_nohelmet(helmet_roi)
            label = 'helmet' if c == 0 else 'no-helmet'
            cv2.putText(img, label, (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if c == 0 else (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            if c == 1:
                frame_no_helmet_found = True

        else:  # number plate (or other class)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # If any no-helmet was detected earlier in this frame, save plate crop + OCR
            if frame_no_helmet_found:
                lp_crop = img[y:y + h, x:x + w]
                if lp_crop.size > 0:
                    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    lp_filename = f"no_helmet_lp_frame{frame_num}_idx{i}.jpg"
                    cv2.imwrite(lp_filename, lp_crop)

                    # OCR license plate
                    plate_text = extract_plate_text(lp_filename)

                    # Save to list
                    no_helmet_data.append({
                        "frame": frame_num,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "lp_image": lp_filename,
                        "plate_number": plate_text
                    })
                    # Avoid duplicating if multiple plates in same frame; reset flag if desired
                    frame_no_helmet_found = False

    # write output video frame
    writer.write(img)

    # show or save frame
    if _can_show:
        try:
            cv2.imshow("Image", img)
            if cv2.waitKey(1) == 27:  # ESC to quit
                break
        except Exception as e:
            # GUI broke mid-run; fallback to saving frames
            _can_show = False
            print("OpenCV GUI broken during runtime; will save frames instead. Error:", e)
            cv2.imwrite(f"frame_{frame_idx}.jpg", img)
    else:
        # save every Nth frame to avoid flooding disk; here we save every 30th
        if frame_idx % 30 == 0:
            cv2.imwrite(f"frame_{frame_idx}.jpg", img)

# --- Save CSV ---
if no_helmet_data:
    df = pd.DataFrame(no_helmet_data)
    df.to_csv("no_helmet_plates_with_text.csv", index=False)
    print("CSV saved with license plate numbers!")
else:
    print("No violations detected. CSV not created.")

writer.release()
cap.release()
cv2.destroyAllWindows()
