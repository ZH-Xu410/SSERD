celeb=${1:-data/MEAD/train/M003}
export PYTHONPATH=.:$PYTHONPATH
python preprocessing/detect.py --celeb ${celeb}
python preprocessing/segment_face.py --celeb ${celeb}
python preprocessing/eye_landmarks.py --celeb ${celeb} --align
python preprocessing/align.py --celeb ${celeb} --faces_and_masks  --images --landmarks