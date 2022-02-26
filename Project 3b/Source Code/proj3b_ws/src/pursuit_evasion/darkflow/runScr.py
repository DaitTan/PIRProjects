from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolov2.cfg", "load": "bin/yolov2.weights"}

tfnet = TFNet(options)

for i in range(9):
	imgcv = cv2.imread("./sample_img/frame000"+str(i)+".jpg")
	result = tfnet.return_predict(imgcv)
	print(result)

	res = [] 
	for listInd, val in enumerate(result):
		if val['label'] == "person":
			if val['confidence'] >= 0.5:
				a = [val['topleft']['x'], val['topleft']['y'],val['bottomright']['x'], val['bottomright']['y'], val['confidence'], val['label']]
				res.append(a)
	print(res)
	print("**************************************************************************")


for i in range(10,14):
	imgcv = cv2.imread("./sample_img/frame00"+str(i)+".jpg")
	result = tfnet.return_predict(imgcv)

	res = [] 
	for listInd, val in enumerate(result):
		if val['label'] == "person":
			if val['confidence'] >= 0.5:
				a = [val['topleft']['x'], val['topleft']['y'],val['bottomright']['x'], val['bottomright']['y'], val['confidence'], val['label']]
				res.append(a)
	print(res)
	print("**************************************************************************")


