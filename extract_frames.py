import cv2, os, tempfile
cap = cv2.VideoCapture(r'C:\Users\tthaker\Downloads\pops_demo_output (13).mp4')
out_dir = os.path.join(tempfile.gettempdir(), 'pops_frames_v13b')
os.makedirs(out_dir, exist_ok=True)
for i in range(270, 370, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join(out_dir, f'frame_{i:04d}.jpg'), frame)
cap.release()
print(f'Saved to {out_dir}')
