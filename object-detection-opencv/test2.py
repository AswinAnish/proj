
import cv2
import argparse
import numpy as np
import math

vid = cv2.VideoCapture(0)

wh = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
ht = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_pos = int(wh / 2)
line_pos2=int(ht/2)
print(line_pos)  
print(ht)
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #print(class_id)
    cent_x=(x+x_plus_w)/2
    cent_y=(y+y_plus_h)/2
    #print(cent_x,cent_y)
    if(cent_x<320):
        if(cent_y>=240):
            print("the person is present is present in 3rd quadrant")
        else:
            print("the person is present is present in 1st quadrant")
    if(cent_x>320):
        if(cent_y<240):
            print("the person is present is present in 2nd quadrant")
        else:
            print("the person is present is present in 4th quadrant")
    #else:
    #   print("the person is present in the middle")
while True:    
    # image = cv2.imread(args.image)
    ret, image = vid.read()

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    cv2.line(image, (line_pos, 0), (line_pos, ht), (0, 0, 0), 1)
    cv2.line(image, (0,line_pos2),(wh,line_pos2), (0, 0, 0), 1)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        if class_ids[i] == 0:
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))


    cv2.imshow("object detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()   
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()

