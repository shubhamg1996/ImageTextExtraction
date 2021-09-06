import cv2
import numpy as np
import pytesseract

def get_data(img,config="--psm 12"):
    #fetch text
    dt=pytesseract.image_to_data(img,output_type="data.frame",config=config)
    
    #filter text detections for only picking words
    dtx=dt[dt['conf']>30][['text','left','top','width','height']]

    if len(dtx) > 0:
        #calculating avg. character width and max text height
        sm_width=0
        ht=[]

        avg_width, max_height = 0, 0

        num_words = len(dtx)

        for index,row in dtx.iterrows():
            try:
                sm_width+=(row['width'])/len(row['text'])
                ht.append(row['height'])
            except :
                # Skip the text object
              
                num_words -= 1
                continue
           
        
        if num_words > 0:
            avg_width = sm_width / num_words

        if ht:
            max_height = max(ht)

        if avg_width == 0:
            # Minimum text width
            # Need to find a better way to avoid hardcoding
            avg_width = 10

        return avg_width, max_height, dtx

    else:
        # No text detected by Pytesseract
        logger.warning('No text detected by Pytesseract')
        return None

def draw_tbox(img,tboxes):
    "returns image to use for detecting space between text to draw borders"
    #create image for border creation
    imgb=np.zeros(img.shape)
    img_h,img_w=img.shape[:2]

    for _,row in tboxes.iterrows():
        x,y,w,h=row['left'],row['top'],row['width'],row['height']
        xmax,ymax=x+w,y+h
        #shifting borderline text boxes to ensure lines being drawn to cover whole content
        if (x==0):
            x=x+1
        elif (xmax)==(img_w):
            xmax=xmax-1
        elif (y==0):
            y=y+1
        elif (ymax)==(img_h):
            ymax=ymax-1    
        #draw tight text boxes 
        cv2.rectangle(imgb,(x+1,y+1),(xmax-1,ymax-1),255,1) 
    return imgb
    
def fetch_space(imgb,direction="row"):
    if direction=='row':
        axis=1
    #get image areas with no text content
    sofdir=np.sum(imgb,axis=axis)
    #list of indexes having no text boxes along specified direction
    idx=[i for i,p in enumerate(sofdir) if p==0]
    #grouping consecutive blanks together
    zeros=[]
    # print(sofdir)
    #consecutive zeros
    cont=[]
    for i in idx:
        if cont:
            print("aaaaaaaa",cont[-1]+1)
            if(i==(cont[-1]+1)):
                cont.append(i)
            else:
                zeros.append(cont)
                cont=[]
                cont.append(i)
        else:
            cont.append(i)
    if cont:
        zeros.append(cont)
    
    return zeros

def filter_lines(zeros, filter_gap=0):
    zerosf=zeros.copy()
    for i in zeros[1:-1]:
        if(len(i)<=filter_gap):
            zerosf.remove(i)
    
    return zerosf

def get_mask(img, get_lines=False, div=3, thresh_val=0,kernel=None):
    """Process image to generate binary mask image containing only the lines of the image."""

    # Thresholding the image
    # THRESH_BINARY_INV = 1, THRESH_OTSU = 8
    # If the threshold value is specified (manual thresholding),
    # then Otsu's Thresholding needs to be turned off, this can
    # be done by setting the parameter to 0. (Default for Otsu is 8)
    
    OTSU_THRESH = 8
    
    if thresh_val > 0:
        OTSU_THRESH = 0
        
    _, img_bin = cv2.threshold(img, thresh_val, 255,
                                      cv2.THRESH_BINARY_INV | OTSU_THRESH)

    # Defining a kernel length
    if kernel is None or kernel == 0:
        hkernel_length = np.array(img).shape[1]//div
        vkernel_length = np.array(img).shape[0]//div
    else:
        hkernel_length=int(kernel *1.02)
        vkernel_length=int(kernel *1.02)

    if hkernel_length == 0 or vkernel_length == 0:
        logger.warning(
            f'Invalid Kernel lengths with kernel={kernel}, hkernel = {hkernel_length} and vkernel = {vkernel_length}. Mask cannot be created')
        
        if get_lines:
            return None, None

        else:
            return None

    # A vertical kernel of (1 X kernel_length), which will detect all the vertical lines from the image.
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vkernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    horiz_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (hkernel_length, 1))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=1)
    vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=1)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, horiz_kernel, iterations=1)
    horizontal_lines_img = cv2.dilate(img_temp2, horiz_kernel, iterations=1)
    
    if get_lines:
        return horizontal_lines_img, vertical_lines_img
    
    else:
        return horizontal_lines_img + vertical_lines_img

if __name__=="__main__":
    img = cv2.imread(r"C:\Users\Shubham Gupta\Downloads\doc_25.jpg",0)
    temp_data = get_data(img)
    avg_width,max_height,tboxes = temp_data
    imgb=draw_tbox(img,tboxes)

    filter_row = 0
    zeros_r=fetch_space(imgb,direction="row")
    # print("a"*10,zeros_r)
    if zeros_r:
        sy=zeros_r[0]
        start_y=sy[round(len(sy)/2)]
        ey=zeros_r[-1]
        end_y=ey[round(len(ey)/2)]
    else:
            # Handle empty zeros
        start_y = 0
        end_y = img.shape[0]

    start_x = 0
    end_x = img.shape[1]
    zeros_r=filter_lines(zeros_r,filter_row)

    img=cv2.threshold(img,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    mask=get_mask(img,kernel=max_height)
    if mask is None:
        img_wl = img
    else:
        img_wl=255-cv2.subtract(cv2.bitwise_not(mask),img)

    img_border = img_wl.copy()
    lines_list = []
    for k in zeros_r:
        lines_list.append(k[round(len(k)/2)])
        cv2.line(img_border,(start_x,k[round(len(k)/2)]),(end_x,k[round(len(k)/2)]),(0),1)
    # print(lines_list)


    dic = {}
    done = []
    for i in range(0,len(lines_list)-1):
        dic[i] = []
        y1 = lines_list[i]
        y2 = lines_list[i+1]
        for j,row in tboxes.iterrows():
            if row["top"] + row["height"] < y2 and row["top"] + row["height"] > y1:
                dic[i].append(row) 
    for i,k in dic.items():
        line = []
        t = []
        for j in k:
            x,y,w,h,text = j["left"],j["top"],j["width"],j["height"],j["text"]
            line.append(x)
            t.append(text)
        arr = np.array(t)
        # print(line,t)
        # print(np.argsort(line))
        print(" ".join(arr[np.argsort(line)]))
        # print([p for p in np.lexsort((line[0]))])
        print("--------new line---------")