import cv2


def is_available(temperature):
    if temperature:
        temperature = str(temperature[0]).split("'")[1][:4]
        try:
            temperature = float(temperature)
            return [True, temperature]
        except ValueError:
            return [False]
    else:
        return [False]


def get_color(color):
    colors = {
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'sky': (255, 255, 0),
        'black': (0, 0, 0),
    }
    return colors[color]
    
    
def draw_bbox(raw_img, bbox, color, label):
    x, y, w, h = bbox
    color = get_color(color)
    
    categories = {
        2: 'mask',
        1: 'unweared_mask',
    }
    
    cv2.rectangle(raw_img, (x, y, w, h), color, 1)
    (tw, th), _ = cv2.getTextSize(categories[label], cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
    cv2.rectangle(raw_img, (x, y-20), (x+tw, y), color, -1)
    cv2.putText(raw_img, categories[label], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(raw_img, (x+w//2, y+h//2), 2, color, 2)
    
    
def draw_below(message, raw_img, frame_height, frame_width, color):
    color = get_color(color)
    (tw, th), _ = cv2.getTextSize(message, 
        cv2.FONT_HERSHEY_COMPLEX, 3, 2)
    cv2.rectangle(
        raw_img, 
        (0, frame_height-100), 
        (frame_width, frame_height-100+th), 
        color, -1)
    cv2.putText(raw_img, message, 
        (frame_width//4, frame_height-100+th//4), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        3, (255, 255, 255), 2, cv2.LINE_AA)
    

def draw_above(message, raw_img, color):
    color = get_color(color)
    (tw, th), _ = cv2.getTextSize(message, 
        cv2.FONT_HERSHEY_COMPLEX, 3, 2)
    cv2.putText(raw_img, message, 
        (100, th+5), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        3, color, 2, cv2.LINE_AA)
    
    
def get_temperature(arduino_serial, frame_width, frame_height, raw_img, bbox):
    _, _, w, h = bbox
        
    if w > 300 and h > 400:
        temperature = arduino_serial.readlines()
        
        if not is_available(temperature)[0]:
            draw_below('Measuring...', 
                raw_img, frame_height, frame_width, 'green')
            return
            
        temperature = is_available(temperature)[1]
            
        if temperature < 30.0:
            draw_below('Please come close.', 
                raw_img, frame_height, frame_width, 'green')
        else:
            draw_below(str(temperature), 
                raw_img, frame_height, frame_width, 'green')
