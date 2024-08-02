# importing the module 
import cv2 
import math
IMG_WIDTH = 1251
iMG_HEIGHT = 1805

def detect_anchor(img, regiions):
	x1 =  regiions[0]
	x2 =  regiions[1]
	y1 =  regiions[2]
	y2 =  regiions[3]
	image = img[x1:x2,y1:y2]
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 50, 150)
	corners = cv2.goodFeaturesToTrack(edges, 25, 0.01, 10)
	print(corners)
	if corners is None:
		print("No Corners Found")
		return img
	for corner in corners:
		x, y = corner[0]
		cv2.circle(img, (int(x)+y1, int(y)+x1), 3, (0, 0, 255), -1)
	return img

def get_coordniates_from_label_studio(x,y,w,h):
	x1 = math.floor((IMG_WIDTH/100)*x)
	x2 = x1 + math.floor((IMG_WIDTH/100)*w)
	y1 = math.floor((iMG_HEIGHT/100)*y)
	y2 = y1 + math.floor((IMG_WIDTH/100)*h)
	return (x1,x2,y1,y2)
	
	
# function to display the coordinates of 
# of the points clicked on the image 
def click_event(event, x, y, flags, params): 

	# checking for left mouse clicks 
	if event == cv2.EVENT_LBUTTONDOWN: 

		# displaying the coordinates 
		# on the Shell 
		print(x, ' ', y) 

		# displaying the coordinates 
		# on the image window 
		font = cv2.FONT_HERSHEY_SIMPLEX 
		cv2.putText(img, str(x) + ',' +
					str(y), (x,y), font, 
					1, (255, 0, 0), 2) 
		cv2.imshow('image', img) 

	# checking for right mouse clicks	 
	if event==cv2.EVENT_RBUTTONDOWN: 

		# displaying the coordinates 
		# on the Shell 
		print(x, ' ', y) 

		# displaying the coordinates 
		# on the image window 
		font = cv2.FONT_HERSHEY_SIMPLEX 
		b = img[y, x, 0] 
		g = img[y, x, 1] 
		r = img[y, x, 2] 
		cv2.putText(img, str(b) + ',' +
					str(g) + ',' + str(r), 
					(x,y), font, 1, 
					(255, 255, 0), 2) 
		cv2.imshow('image', img) 

# driver function 
if __name__=="__main__":
	# cv2.namedWindow("image",cv2.WINDOW_GUI_EXPANDED)
	img = cv2.imread('1.jpg', 1) 

	# First Anchor
	# x1,x2,y1,y2 = get_coordniates_from_label_studio(9.70,9.01,3.43,2.54)
	# img = detect_anchor(img,[y1,y2,x1,x2])
	# # img = detect_anchor(img,[9,13,9,13])
	
	# # # Second Anchor
	# x1,x2,y1,y2 = get_coordniates_from_label_studio(71.82,9.01,3.25,2.33)
	# img = detect_anchor(img,[y1,y2,x1,x2])
	
	# # img = detect_anchor(img,[140,220,880,970])
	# # img = detect_anchor(img,[70,74,9,12])
	
	# # # Third Anchor
	# x1,x2,y1,y2 = get_coordniates_from_label_studio(94.14,11.88,3.12,2.54)
	# img = detect_anchor(img,[y1,y2,x1,x2])

	# img = detect_anchor(img,[212,245,1175,1220])
	# img = detect_anchor(img,[94,97,11,13])
	
	cv2.imshow('image', img)

	# setting mouse handler for the image 
	# and calling the click_event() function 
	cv2.setMouseCallback('image', click_event) 

	# wait for a key to be pressed to exit 
	cv2.waitKey(0) 

	# close the window 
	cv2.destroyAllWindows() 