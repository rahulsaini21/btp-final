import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.widgets import Slider
from math import hypot


# selectroi = False


# play_video = "./Videos/video_1.mp4"
# show_crop = True
# n_clusters = 5


# play_video = "./Videos/disAlign_100.mp4"
# cal_video = "./Videos/disAlign_100.mp4"
# show_crop = False
# n_clusters = 3
# (33,53,80)


# play_video = "./Videos/final_5mm.mp4"
# cal_video = "./Videos/final_5mm.mp4"
# show_crop = False
# n_clusters = 3

# play_video = "./Videos/final_10mm.mp4"
# cal_video = "./Videos/final_10mm.mp4"
# show_crop = False
# n_clusters = 3

# play_video = "./Videos/dif_10_rot.mp4"
# cal_video = "./Videos/dif_10_cal.mp4"
# show_crop = False
# n_clusters = 4

# play_video = "./Videos/dif_5_rot.mp4"
# cal_video = "./Videos/dif_5_cal.mp4"
# show_crop = False
# n_clusters = 4



# control2_2.5_2.3_7.5_7.mp4
# cal_video = "./Videos/control2_2.5_2.3_7.5_7_cut.mp4"
# play_video = "Videos/control2_2.5_2.3_7.5_7_rot.mp4"
# show_crop = True
# n_clusters = 4



# play_video = "E:/SEMESTERS/Semester 7th/btp-git/BTP_1/GITHUB/Videos/control2_2.5_2.3_7.5_7_rot.mp4"
# cal_video = "E:/SEMESTERS/Semester 7th/btp-git/BTP_1/GITHUB/Videos/control2_2.5_2.3_7.5_7_rot.mp4"
# show_crop = False
# n_clusters = 3

vid_name = "E:/SEMESTERS/Semester 7th/btp-videos/Rahul_Mansi/new/C0028.mp4"
# vid_name = "E:/SEMESTERS/Semester 7th/btp-videos/Chandigarh videos/00006.MTS"
play_video = vid_name
cal_video = vid_name
show_crop = False
n_clusters = 4

alpha = 1  # Contrast factor (greater than 1 increases contrast)
beta = 0    # Brightness factor (usually 0 for contrast adjustment)
mb_val = 5
gb_val = 5
gb_i = 0
resol = 180
scale_up = 1
blur_level = 1
frame_index = 0

cap = cv2.VideoCapture(play_video)
# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Read the first frame
ret, frame = cap.read()




# Create a Matplotlib figure
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
plt.title('Video Frame')
plt.axis('off')
frame_display = plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Create a slider to adjust blur level
ax_blur_slider = plt.axes([0.15, 0.1, 0.8, 0.05])
blur_slider = Slider(ax_blur_slider, 'Blur Level', 1, 100, valinit=1, valstep=2)

ax_median_slider = plt.axes([0.15, 0.15, 0.8, 0.05])
median_slider = Slider(ax_median_slider, 'median blue', 1, 100, valinit=1, valstep=2)

ax_contrast_a = plt.axes([0.15, 0.20, 0.8, 0.05])
contrast_a_slider = Slider(ax_contrast_a, 'contrast a', 0, 10, valinit=0, valstep=0.2)

ax_contrast_b = plt.axes([0.15, 0.25, 0.8, 0.05])
contrast_b_slider = Slider(ax_contrast_b, 'contrast b', -127, 127, valinit=0, valstep=5)

def update_blur(val):
    global blur_level
    global frame_index
    blur_level = int(val)
    update_frame(frame_index)

blur_slider.on_changed(update_blur)

def update_median(val):
    global mb_val
    global frame_index
    mb_val = int(val)
    update_frame(frame_index)

median_slider.on_changed(update_median)

def update_contrast_a(val):
    global alpha
    global frame_index
    alpha = int(val)
    update_frame(frame_index)

contrast_a_slider.on_changed(update_contrast_a)

def update_contrast_b(val):
    global beta
    global frame_index
    beta = int(val)
    update_frame(frame_index)

contrast_b_slider.on_changed(update_contrast_b)


# frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
# frame = cv2.medianBlur(frame, mb_val)
# frame = cv2.GaussianBlur(frame, (gb_val, gb_val), gb_i)

def update_frame(frame_index):
    # global blur_level
    # Set the frame position in the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    # Read the frame at the specified position
    ret, frame = cap.read()
    # Apply Gaussian blur to the frame
    print("alpha: ", alpha)
    frame = cv2.GaussianBlur(frame, (blur_level, blur_level), 0)
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    frame = cv2.medianBlur(frame, mb_val)
    frame = cv2.GaussianBlur(frame, (gb_val, gb_val), gb_i)
    # Display the frame
    frame_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fig.canvas.draw_idle()

# Create a slider to select the frame
ax_frame_slider = plt.axes([0.15, 0.30, 0.8, 0.05])
frame_slider = Slider(ax_frame_slider, 'Frame', 0, total_frames - 1, valinit=0, valstep=1, valfmt='%d')



def on_frame_change(val):
    global frame_index
    frame_index = int(val)
    update_frame(frame_index)

frame_slider.on_changed(on_frame_change)

# Show the Matplotlib plot
x = 0
y = 0
def onclick(event):
    global x
    global y
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        print(f"Clicked pixel coordinates: ({x}, {y})")

# Connect the mouse click event to the function
plt.gcf().canvas.mpl_connect('button_press_event', onclick)


plt.show()

cap.release()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# color picker
# color picker
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def elcu_dis(x1,y1,x2, y2):
    return np.sqrt((x1-x2)*(x1-x2)+ (y1-y2)*(y1-y2))

unit_distance = int(input("calibration_distance: "))
cm_pixel = np.array([])
cali_x = 0
cali_y = 0

def cal_line(event, x, y, flags, param):
    global cali_x, cali_y, cm_pixel
    check, factor = param
    if event == cv2.EVENT_LBUTTONDOWN:
        cali_x = x
        cali_y = y

    elif event == cv2.EVENT_LBUTTONUP:
        pixel_distance = elcu_dis(cali_x, cali_y, x, y)
        print("pixel_distance ",cali_x, cali_y, x, y, pixel_distance)
        if check:
            print(factor*pixel_distance)
        else:
            scale = abs(unit_distance/pixel_distance)
            cm_pixel = np.append(cm_pixel, scale)


cap = cv2.VideoCapture(cal_video) #name of image
ret, frame = cap.read()
# roi = cv2.selectROI(frame)
cv2.namedWindow('calibrate')
cv2.setMouseCallback('calibrate',cal_line, param = (False, 0))


while(1):
    ret, frame = cap.read()
    # frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    cv2.putText(frame, "Mark calibration distances", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, "Press n to move to next frame", (100,125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, "Press c to end step", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # c_frame = frame[int(roi[1]):int(roi[1]+roi[3]), 
    #                   int(roi[0]):int(roi[0]+roi[2])]
    
    # aspect_ratio = c_frame.shape[1] / c_frame.shape[0]
    # print("roi:", roi)
    # print("aspect ratio:", aspect_ratio)
    # c_frame = cv2.resize(c_frame, (int(aspect_ratio*360), 360), fx=2, fy=2)
    # c_frame = cv2.resize(c_frame
    
    # cv2.namedWindow('calibrate')
    # cv2.setMouseCallback('calibrate',cal_line, param = (False, 0))


    cv2.imshow('calibrate', frame)
    key = cv2.waitKey(0) & 0xFF
    # Press 'c' to end the loop
    if key == ord('c'):
        break
    # Press 'n' to move to the next frame
    elif key == ord('n'):
        continue


cv2.destroyAllWindows()
f_scale = np.average(cm_pixel)
print("scaling factor: ", f_scale)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#check calibration
cap = cv2.VideoCapture(cal_video) #name of image
ret, frame = cap.read()
cv2.namedWindow('verify_c')
cv2.setMouseCallback('verify_c',cal_line, param = (True, f_scale))
while(1):
    ret, frame = cap.read()
    # frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    cv2.putText(frame, "Check calibrated distance", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, "Press c to end step", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imshow('verify_c', frame)
    key = cv2.waitKey(0) & 0xFF
    # Press 'c' to end the loop
    if key == ord('c'):
        break
    # Press 'n' to move to the next frame
    elif key == ord('n'):
        continue
cv2.destroyAllWindows()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


colRange = np.array([[1000, 1000, 1000], [0, 0, 0]])

r_mean = 0
g_mean = 0
b_mean = 0

def mouseRGB(event,x,y,flags,param):
    global r_mean, g_mean, b_mean
    mframe, colorRange, rx, ry = param
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = mframe[y,x,0]
        colorsG = mframe[y,x,1]
        colorsR = mframe[y,x,2]
        # colors = mframe[y,x]
        
        hsv_value= np.uint8([[[colorsB ,colorsG,colorsR ]]])
        hsv = cv2.cvtColor(hsv_value,cv2.COLOR_BGR2HSV)
        print ("HSV : " ,hsv)
        colorRange[0][0] = min(colorRange[0][0], hsv[0][0][0])
        colorRange[0][1] = min(colorRange[0][1], hsv[0][0][1])
        colorRange[0][2] = min(colorRange[0][2], hsv[0][0][2])
        colorRange[1][0] = max(colorRange[1][0], hsv[0][0][0])
        colorRange[1][1] = max(colorRange[1][1], hsv[0][0][1])
        colorRange[1][2] = max(colorRange[1][2], hsv[0][0][2])

        r_mean = (colRange[0][0]+colRange[1][0])/2
        g_mean = (colRange[0][1]+colRange[1][1])/2
        b_mean = (colRange[0][2]+colRange[1][2])/2

        print("Coordinates of pixel: X: ",x,"Y: ",y)
        print("Coordinates of pixel: X: ",rx+x,"Y: ",ry+y)
        print("colRange:", colRange)
        

        


cap = cv2.VideoCapture(play_video) #name of image

ret, frame = cap.read()



# while(1):
#     ret, frame = cap.read()
#     frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
#     cv2.namedWindow('full')
#     cv2.setMouseCallback('full',mouseRGB, param = (frame, colRange, 0, 0))
#     cv2.imshow('full', frame)
#     key = cv2.waitKey(0) & 0xFF
#     print('inside')
#     # Press 'c' to end the loop
#     if key == ord('c'):
#         break
#     # Press 'n' to move to the next frame
#     elif key == ord('n'):
#         continue
# cv2.destroyAllWindows()
#Do until esc pressed

# if selectroi:
cv2.putText(frame, "Select Region Of Interest", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
cv2.putText(frame, "Press n to move to next frame", (100,125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
cv2.putText(frame, "Press c to end step", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
roi = cv2.selectROI(frame)
# print(roi)
while(1):
    ret, frame = cap.read()
    # ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    # Apply the contrast adjustment
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    frame = cv2.medianBlur(frame, mb_val)
    frame = cv2.GaussianBlur(frame, (gb_val, gb_val), gb_i)
    # ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

    c_frame = frame[int(roi[1]):int(roi[1]+roi[3]), 
                      int(roi[0]):int(roi[0]+roi[2])]
    
    aspect_ratio = c_frame.shape[1] / c_frame.shape[0]
    # c_frame = cv2.resize(c_frame, (int(aspect_ratio*resol), resol), fx=scale_up, fy=scale_up)

    cv2.namedWindow('mouseRGB')
    cv2.setMouseCallback('mouseRGB',mouseRGB, param = (c_frame, colRange, roi[0], roi[1]))
    
    # colorRange[0][1] = min(colorRange[0][1], hsv[0][0][1])
    # colorRange[0][2] = min(colorRange[0][2], hsv[0][0][2])
    # colorRange[1][0] = max(colorRange[1][0], hsv[0][0][0])
    # colorRange[1][1] = max(colorRange[1][1], hsv[0][0][1])
    # colorRange[1][2] = max(colorRange[1][2], hsv[0][0][2])
    
    print("printing circlesssssssssss:")
    cv2.circle(c_frame, (0, 0), 50, (r_mean, g_mean, b_mean), 5)
    # cv2.circle(c_frame, (100, 100), 50, (0, 0, 255), 5)

    cv2.putText(frame, "Pick Color to Track", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, "Press n to move to next frame", (100,125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, "Press c to end step", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imshow('mouseRGB', c_frame)
    # cv2.imshow(c_frame)
    
    
    key = cv2.waitKey(0) & 0xFF
    
    # Press 'c' to end the loop
    if key == ord('c'):
        break
    # Press 'n' to move to the next frame
    elif key == ord('n'):
        continue
#if esc pressed, finish.

print(colRange)
cv2.destroyAllWindows()




lower = colRange[0]-50
upper = colRange[1]+50            



#  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# end_x finder
# end_x finder
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
cap = cv2.VideoCapture(play_video) #name of image
ret, frame = cap.read()
max_X = np.array([])
line_ys = [0,0]
while True:
    success, frame = cap.read() 
    if(success == False):
            break


    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    # Apply the contrast adjustment
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    frame = cv2.medianBlur(frame, mb_val)
    frame = cv2.GaussianBlur(frame, (gb_val, gb_val), gb_i)

    # ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

    c_frame = frame[int(roi[1]):int(roi[1]+roi[3]), 
                        int(roi[0]):int(roi[0]+roi[2])]
    
    cc_frame = c_frame.copy() 

    img = cv2.cvtColor(c_frame, cv2.COLOR_BGR2HSV) # Converting BGR image to HSV format

    mask = cv2.inRange(img, lower, upper) # Masking the image to find our color

    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image

    # Finding position of all contours
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 5:
                x, y, w, h = cv2.boundingRect(mask_contour)
                # cv2.rectangle(c_frame, (x, y), (x + w, y + h), (0, 0, 255), 3) #drawing rectangle
                max_X = np.append(max_X, x+w)
                line_ys[0] = y-10
                line_ys[1] = y+10
                print("draw")
                cv2.line(c_frame, (x+w, y-10), (x+w, y+10), (0,0,255), 3)

    
    cv2.putText(frame, "Caculating blade range in ROI", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    # cv2.putText(frame, "Press n to move to next frame", (100,125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, "Press c to end step", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imshow("window", frame) # Displaying webcam image

    if cv2.waitKey(1) & 0xff == ord('c'):
        break

# print(max_X)
end_x = max_X.max()
# print("mean: ", end_x)
cv2.destroyAllWindows()

#  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx



# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# contour
# contour
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

verticle = []
verticle_ref = []
verticle_ref_corner = []
all_corners = []
cap = cv2.VideoCapture(play_video) #name of image

# roi = cv2.selectROI(frame)

line_range = [end_x-80, end_x+80]

# tracker = cv2.TrackerCSRT_create()  # You can try different trackers like KCF, MOSSE, etc.
tracker = cv2.legacy.TrackerCSRT_create()

# Read the first frame
ret, frame = cap.read()
# frame = cv2.resize(frame, (720 , 720))

# Select a ROI (Region of Interest) to track
bbox = cv2.selectROI(frame)
tracker.init(frame, bbox)

# def inside()
y_ref = frame.shape[0]

while True:

    ####################################################################################
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (720 , 720))
    if not ret:
        break
    
    # Update tracker
    ret, bbox = tracker.update(frame)
    
    # Draw bounding box
    if ret:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        
        x_ref = int(bbox[0] + bbox[2])
        y_ref = int(bbox[1] + bbox[3])

        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    ####################################################################################


    # success, frame = cap.read() # Reading webcam footage
    if(ret == False):
            break

    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    # Apply the contrast adjustment
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    frame = cv2.medianBlur(frame, mb_val)
    frame = cv2.GaussianBlur(frame, (gb_val, gb_val), gb_i)
    # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


    c_frame = frame[int(roi[1]):int(roi[1]+roi[3]), 
                        int(roi[0]):int(roi[0]+roi[2])]
    
    cc_frame = c_frame.copy()
    
    
    # img = cv2.cvtColor(c_frame, cv2.COLOR_BGR2HSV) # Converting BGR image to HSV format

    # mask = cv2.inRange(img, lower, upper) # Masking the image to find our color

    # mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image

    # # Finding position of all contours
    # if len(mask_contours) != 0:
    #     print('length_contours ', len(mask_contours))
    #     for mask_contour in mask_contours:
    #         if cv2.contourArea(mask_contour) > 5:
    #             x, y, w, h = cv2.boundingRect(mask_contour)
    #             cv2.rectangle(c_frame, (x, y), (x + w, y + h), (0, 0, 255), 3) #drawing rectangle
    #             # verticle =  np.append(verticle, ([x+w, y+h/2]))
    #             if line_range[0] <= x+w:
    #                 verticle.append([x+w, int(y+h/2)])
    #                 verticle_ref.append([x+w, int(y_ref - (y+h/2))])

    #                 # verticle.append(int(y+h/2+roi[1]))
    #                 # verticle_ref.append(int(y_ref - (y+h/2)+ roi[1]))

    #                 cv2.circle(c_frame, (int(x+w),int(y+h/2)), 1, (255, 0, 0), 5)
    #                 cv2.circle(frame, (x_ref, y_ref), 10, (0, 255, 0), 5)
    #                 # print( "store points ", int(x+w),int(y+h/2))
    #                 # print( "y-ref:", int(y_ref))
    #                 # print( "y-abs: ", int((y+h/2)+ roi[1]))
    #                 # print( "rel-verticle: ",  int(y_ref - (y+h/2)+ roi[1]) )

    #                 # key = cv2.waitKey(0) & 0xFF
    #                 # if key == ord('n'):
    #                 #         continue

                
    #             # print("points ")
    #             # print(verticle)
    #             # for i in verticle:
    #             #     cv2.circle(c_frame, (int(i[0]),int(i[1])), 10, (255, 0, 0), 5)


    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    print("corner detections:")
    operatedImage = cv2.cvtColor(cc_frame, cv2.COLOR_BGR2GRAY)

    # Modify the data type and apply cv2.cornerHarris
    operatedImage = np.float32(operatedImage)
    dest = cv2.cornerHarris(operatedImage, 7, 7, 0.06)

    # Results are marked through the dilated corners
    # print(dest)
    dest = cv2.dilate(dest, None)
    # print(dest)
    max_value = dest.max()
    max_loc = np.where(dest == max_value)
    verticle_ref_corner.append([max_loc[1][0], int(y_ref - max_loc[0][0])])
    # Find and draw the corners on the original frame
    c_frame[dest > 0.99 * dest.max()] = [0, 255, 0]

    # Display the frame with corner points
    cv2.imshow('Video with Corners', frame)


    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Change 50 to the no of corners we want to detect 
    # gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
    # corners = cv2.goodFeaturesToTrack(gray, 10, 0.01, 20)   

    # if corners is not None:
    #     corners = np.int0(corners)

    #     for corner in corners:
    #         x, y = corner.ravel()
    #         if line_range[0] <= x+w:
    #             cv2.circle(c_frame, (x, y), 3, (0, 0, 255), -1)
    #             all_corners.append((x,y))
    #     print(corners)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # cv2.imshow("mask image", mask) # Displaying mask image

    # c_frame = cv2.resize(c_frame, (720, 720))
    # c_frame = cv2.resize(c_frame, fx=2, fy=2)

    cv2.line(c_frame, (int(line_range[0]), int(line_ys[0])), (int(line_range[0]), int(line_ys[1])), (255, 0, 0), 3) #drawing rectangle
    cv2.line(c_frame, (int(line_range[1]), int(line_ys[0])), (int(line_range[1]), int(line_ys[1])), (255, 0, 0), 3) #drawing rectangle

    
    cv2.putText(frame, "Storing blade X, Y cordinate of Tips", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    # cv2.putText(frame, "Press n to move to next frame", (100,125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, "Press c to end step", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    if(show_crop):
        # c_frame = cv2.resize(c_frame, resol, fx=scale_up, fy=scale_up)
        c_frame = cv2.resize(c_frame, (int(aspect_ratio*resol),resol), fx=scale_up, fy=scale_up)
        cv2.imshow("window", c_frame) # Displaying webcam image
    else:
        cv2.imshow("window", frame) # Displaying webcam image


    # key = cv2.waitKey(0) & 0xFF
    
    if cv2.waitKey(1) & 0xff == ord('c'):
        break
    # Press 'c' to end the loop
    # if key == ord('c'):
    #     break
    # Press 'n' to move to the next frame
    # elif key == ord('n'):
    #     continue
    # else:
    #     continue
cv2.destroyAllWindows()
###########################################################################################
# y_num = len(verticle)
# print("y_num:", y_num)
# y_num_cord = range(1, y_num + 1)
# verticle_y = [row[1] for row in verticle]
# verticle_rel_y = [row[1] for row in verticle_ref]

# # print(verticle)
# # print(verticle_ref)
# plt.plot(y_num_cord, verticle_y, label='Array 1')
# plt.plot(y_num_cord, verticle_rel_y, label='Array 2')

# # Adding labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Two Arrays Plot')

# # Adding legend
# plt.legend()

# # Display the plot
# plt.show()
#############################################################################################
#####################################################################
# slider
fig = plt.figure()
ax = fig.subplots()
plt.subplots_adjust(bottom=0.25)


#define slider
#property

#####################################################################

# print("verticle", verticle)

data_to_plot_x = [row[0] for row in verticle_ref_corner]
# data_to_plot_y = [row[1] for row in verticle_ref]

data_to_plot_y = [row[1] for row in verticle_ref_corner]


p = ax.plot(data_to_plot_x, marker='o',color='blue', linestyle='-', label='x-cordinate')
p = ax.plot(data_to_plot_y, marker='o',color='red', linestyle='--', label='y-cordinate')
# plt.title(f'Graph of Column {column_to_plot + 1}')
# plt.xlabel('Row Index')
# plt.ylabel(f'Column {column_to_plot + 1} Value')

#####################################################################
# Perform Fourier Transform
fft_result = np.fft.fft(data_to_plot_x)

# Keep only the low-frequency components (e.g., remove high-frequency noise)
num_components_to_keep = int(len(fft_result)/3)
fft_result_copy = fft_result.copy()
fft_result_copy[ num_components_to_keep: -num_components_to_keep] = 0

# Inverse Fourier Transform to reconstruct the signal
smoothed_y = np.fft.ifft(fft_result_copy)

# Plot the original and smoothed signals
p, = ax.plot(np.real(smoothed_y), marker='o',color='green', label='Smoothed Signal', linewidth=2)
# print("pppp ", p)
# plt.show()
#####################################################################


ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03]) #x, y, w, h
ax_slide2 = plt.axes([0.25, 0.05, 0.65, 0.03]) #x, y, w, h
s_factor = Slider(ax_slide, 'Smoothing factor', valmin = 1, valmax = 50, valinit=2, valstep=0.1)
off_factor = Slider(ax_slide2, 'offset factor', valmin = -10, valmax = 10, valinit=0, valstep=1)

#udate the plot
def update(val):
    global smoothed_y
    current_v = s_factor.val
    current_off = off_factor.val
    fft_result = np.fft.fft(data_to_plot_x) 
    num_components_to_keep = int(len(fft_result)/current_v)
    fft_result[ num_components_to_keep: -num_components_to_keep] = 0
    smoothed_y = np.fft.ifft(fft_result)
    add_z = np.ones(abs(current_off))
    if(current_off>0):
        smoothed_y = np.concatenate((add_z*smoothed_y[0], smoothed_y[:-current_off]))
    if(current_off<0):
        add_z = add_z*smoothed_y[-1]
        smoothed_y = np.concatenate((smoothed_y[-current_off:], add_z ))
    p.set_ydata(smoothed_y)
    print(smoothed_y)   
    fig.canvas.draw()
    

s_factor.on_changed(update)
off_factor.on_changed(update)
plt.legend()
ax.legend()
plt.show()
print("smoothed_y new ", smoothed_y)
def find_peak_points(arr, smooth):
    peaks = []

    for i in range(1, len(smooth) - 1):
        if smooth[i - 1] <= smooth[i] >= smooth[i + 1]:
            if i<len(arr):
                peaks.append([1,arr[i][1]*f_scale])

    return peaks

print("len ",len(verticle_ref_corner), len(smoothed_y))
verticle = find_peak_points(verticle_ref_corner, smoothed_y)

def cluster(arr, n_cluster):
    X = np.array(arr)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    print("center ", np.sort(cluster_centers))

    # Plot the clustered points
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(n_cluster):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], label=f'Cluster {i}')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', s=200, label='Centroids')
    # plt.legend()
    # plt.xlabel()
    plt.ylabel('Distance in cm')
    plt.title('K-means Clustering')
    plt.show()

# print()

cluster(verticle, n_clusters)
# cluster(all_corners, n_clusters)

