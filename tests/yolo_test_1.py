from ultralytics import YOLO

model = YOLO('yolo26n.pt')

results = model('Test_pictures/IMG_7534-scaled_868x579.jpg')
results[0].show()