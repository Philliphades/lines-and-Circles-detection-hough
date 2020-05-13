# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:15:45 2019
by:Phuocnguyen
@author: HadesSecurity
"""
"""----------------------------------------------------------------------------
                BTL MÔN NHẬP MÔN XỬ LÝ ẢNH-NHÓM 3-16DDS06031
-------------------------------------------------------------------------------
     Ứng dụng phát hiện đoạn thẳng và hình tròn bằng phép biến đổi Hough
-------------------------------------------------------------------------------
 - Yêu cầu chung:
    + Tổ chức chương trình khá hợp lý, có hàm main và chú thích đầy đủ cho file, 
    hàm và các khối lệnh quan trọng
    + Chương trình chạy được, đúng nội dung của bài tập lớn
    + Có giao diện đồ họa
    + Có tên môn học, tên các thành viên, tên ứng dụng, …
    + Có ít nhất 2 khung hình: ảnh gốc, ảnh đang xử lý.
    + Hiển thị các thông tin ảnh: tên ảnh, kích thước, loại ảnh, cường độ xám 
    tại vị trí con trỏ chuột, ...
    + Các chức năng: mở ảnh để xử lý (open, không sử dụng địa chỉ cố định của
    ảnh trong mã nguồn), lưu ảnh sau khi xử lý (save), ...
-------------------------------------------------------------------------------
Hough Line Transform
    -Biến đổi dòng Hough là một biến đổi được sử dụng để phát hiện các đường thẳng.
    -Để áp dụng Transform, trước tiên, một quá trình tiền xử lý phát hiện cạnh là mong muốn.
    -Có 2 dạng: Trong hệ tọa độ Descartes: Tham số: (m, b).
               Trong hệ tọa độ cực: Tham số: (r, θ)
    -Một phương trình đường có thể được viết là: r=xcosθ+ysinθ
    -mỗi điểm (x0, y0), chúng ta có thể định nghĩa họ các dòng đi qua điểm đó là:
        rθ=x0⋅cosθ+y0.sinθ
    -Có nghĩa là mỗi cặp (rθ, θ) đại diện cho mỗi dòng đi qua (x0, y0).
    -Một đường có thể được phát hiện bằng cách tìm số giao điểm giữa các đường cong.
    Nhiều đường cong giao nhau có nghĩa là đường được biểu thị bởi giao điểm đó 
    có nhiều điểm hơn. Nói chung, chúng ta có thể xác định ngưỡng của số lượng 
    giao cắt tối thiểu cần thiết để phát hiện một dòng.
    -Nó theo dõi giao điểm giữa các đường cong của mọi điểm trong ảnh. 
    Nếu số lượng giao điểm vượt quá ngưỡng nào đó, thì nó khai báo nó là một
    đường có các tham số (, rθ) của điểm giao nhau.
    OpenCV thực hiện hai loại Biến đổi dòng Hough:
        Biến đổi Hough tiêu chuẩn(HoughLines())
        Biến đổi đường thẳng xác suất( HoughLinesP())
-------------------------------------------------------------------------------
Hough Circle Transform
    -Biến đổi Hough Circle hoạt động theo cách gần giống với Biến đổi Hough Line
    được giải thích trong hướng dẫn trước.
    -Trong trường hợp phát hiện dòng, một dòng được xác định bởi hai tham số (r,).
    Trong trường hợp vòng tròn, chúng ta cần ba tham số để xác định một vòng tròn:
                        C:(xcenter,ycenter,r)
    +trong đó (xcenter, ycenter) xác định vị trí trung tâm và r là bán kính, 
    cho phép chúng ta xác định hoàn toàn một vòng tròn.
    -Để đạt hiệu quả, OpenCV thực hiện một phương pháp phát hiện hơi phức tạp hơn
    so với Hough Transform tiêu chuẩn: Phương pháp gradient Hough, 
    được tạo thành từ hai giai đoạn chính. 
    Giai đoạn đầu tiên bao gồm phát hiện cạnh và tìm các trung tâm vòng tròn có thể
    và giai đoạn thứ hai tìm thấy bán kính tốt nhất cho mỗi trung tâm ứng cử viên.
-------------------------------------------------------------------------------    
"""
#import classifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split #Tách các mảng hoặc ma trận thành các tập con ngẫu nhiên và kiểm tra

#import os
import math
import numpy as np
import glob #Tìm kiếm các tập tin có tên thoả mãn với điều kiện cho trước
from tkinter import *
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as tkFileDialog
import cv2
import win32gui
from tkinter import Frame, Tk, BOTH, Menu,Label,Button,Entry
from tkinter.filedialog import Open,asksaveasfilename
#from tkinter import filedialog

#hàm mở file    
def openfile():
        ftypes = [('Python files', '*.*'), ('All files', '*')] #Loại file
        dlg = Open(filetypes = ftypes) 
        fl = dlg.show() 
        global f,frame,labelsize
        if fl != '':
            im = readFile(fl)
            f = fl
            show(im)
            
            im = cv2.imread(f)
            
            #Lấy kích thước
            labelsize = Label(frame,text = "",font="Times 18 bold italic")
            labelsize.place(x=560,y=195)
            height,width, channels = im.shape 
            stringsize="-Kích thước ảnh: " + str(height) + "x" + str(width)
            labelsize.config(text=stringsize)
            print (stringsize)           
            
#đọc file ảnh            
def readFile(fl):
    im = Image.open(fl).resize((500,300))
    return im;
#hiển thị kqảnh    
def show(im):
    im2 = ImageTk.PhotoImage(im)
    global label21
    label21 = Label(width=500,height=300,image = im2)
    label21.image = im2
    label21.place(x=200, y=330)

def show_kq(im):
    im3 = ImageTk.PhotoImage(im)
    global label21
    label21 = Label(width=500,height=300,image = im3)
    label21.image = im3
    label21.place(x=750, y=330)

    
def executeline():  
    global frame,label28,button23
#tạo button
    button21 = Button(frame,width=12,text = "Load",fg = "Dark Cyan",
                     font="Times 14 bold",bg='Azure2',command=openfile)
    button21.place(x=50,y=250)
    button22 = Button(frame,width=12,text = "Xử lý",fg = "Dark Cyan",
                     font="Times 14 bold",bg='Azure2',command=xuly0)
    button22.place(x=50,y=300)
     #tạo label
    label02 = Label(frame,fg = "#FF3399",text = "Ảnh chưa xử lý ",font="Times 14 bold")
    label02.place(x=387,y=290)
    label03 = Label(frame,fg = "#FF3399",text = "Kết quả sau khi xử lý",font="Times 14 bold")
    label03.place(x=895,y=290)
    
#hàm nút xử lý đường thẳng 
def xuly0():
    
    src = cv2.imread(f) #đọc ảnh
    #Đầu tiên phát hiện các biên bằng Canny
    dst = cv2.Canny(src, 50, 200) 
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR) #Chuyển về dạng xám

    if True: # HoughLinesP
        # Phất hiện các đường thẳng Hough bằng Canny
        lines = cv2.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)       
        #   Với các đối số:
        # dst: kết quả phát hiện cạnh bằng Canny, hình ảnh ở dạng xám
        # lines: vector lưu giữ các tham số (r,θ) của các đường được phát hiện
        # rho: độ phân giải của r (pixel): 1 pixel
        # theta: độ phân giải của θ (radian): math.pi/180
        # threshold: số giao điểm tối thiểu để phát hiện đường thẳng
        # minLinLength: số giao điểm tối thiểu tạo thành một đường thẳng (50),
        #nếu nhỏ hơn thì bỏ qua
        # minLinGap: khoảng cách tối đa giữa 2 điểm được xét trong cùng 1 đường:10
#        
        #hiển thị kết quả bằng cách vẽ các đường
        a,b,c = lines.shape
        for i in range(a):
            cv2.line(cdst,(lines[i][0][0],lines[i][0][1]),
                         (lines[i][0][2], lines[i][0][3]), 
                         (0, 0, 255), 3, cv2.LINE_AA)

    else:    # HoughLines
        lines = cv2.HoughLines(dst, 1, math.pi/180.0, 50, np.array([]), 0, 0)
    # Với các đối số:
    # dst: kết quả phát hiện cạnh bằng Canny, hình ảnh ở dạng xám
    # lines: vector lưu giữ các tham số (r,θ) của các đường được phát hiện
    # rho: độ phân giải của r (pixel): 1 pixel
    # theta: độ phân giải của θ (radian): math.pi/180
    # threshold: số giao điểm tối thiểu để phát hiện đường thẳng: 50
    # srn and stn: tham số mặc định là 0
        
        #hiển thị kết quả bằng cách vẽ các đường
        if lines is not None:
            a,b,c = lines.shape
            for i in range(a):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0, y0 = a*rho, b*rho
                pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
                pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imwrite("done.jpg",cdst)
    im3 = Image.open("done.jpg").resize((500,300))
    im3 = ImageTk.PhotoImage(im3)
    global label28
    label28 = Label(image = im3)
    label28.image = im3
    label28.place(x=730,y=330)
   #lưu kết quả 
    def luu():          
        cv2.imwrite("Luu_KQ.jpg",cdst)
    
    button23 = Button(frame,width=12,text = "Save",fg = "Dark Cyan",
                     font="Times 14 bold",bg='Azure2',command=luu)
    button23.place(x=50,y=350)
    

def circle_execute():      
    global frame 
#    global frame3 
#    frame3 = Frame(frame,height=520,width=1050)
#    frame3.place(x=200,y=330)
    button21 = Button(frame,width=12,text = "Load",fg = "#33CC00",
                     font="Times 14 bold",bg='Thistle2',command=openfile)
    button21.place(x=50,y=250)
    button22 = Button(frame,width=12,text = "Xử lý",fg = "#33CC00",
                     font="Times 14 bold",bg='Thistle2',command=xuly_circle)
    button22.place(x=50,y=300)
    
    label02 = Label(frame,fg = "#FF3399",text = "Ảnh chưa xử lý ",font="Times 14 bold")
    label02.place(x=387,y=290)
    label03 = Label(frame,fg = "#FF3399",text = "Kết quả sau khi xử lý",font="Times 14 bold")
    label03.place(x=895,y=290)

    #hàm sử lý hình tròn
def xuly_circle():
    #Hàm dự đoán vật liệu
    def predictMaterial(roi):
        # tính toán véc tơ for vùng quan tâm
        hist = calcHistogram(roi)    
        # dự đoán loại vật liệu
        s = clf.predict([hist])    
        # return loại vật liệu dự đoán
        return Material[int(s)]
    
    #Hàm gọi Histogram
    def calcHistogram(img):
        # create mask
#      xuất ra:  ndarray Mảng số không với hình dạng, dtype và thứ tự đã cho
        m = np.zeros(img.shape[:2], dtype="uint8")#Trả về một mảng mới có hình dạng và kiểu đã cho, chứa đầy các số không.
        (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
        cv2.circle(m, (w, h), 60, 255, -1)    
        # calcHist expects a list of images, color channels, mask, bins, ranges
        #calcHist mong đợi một danh sách các hình ảnh, kênh màu, mặt nạ, thùng, phạm vi
        h = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 256, 0, 256, 0, 256])    
        # return normalized "flattened" histogram
        #trả về biểu đồ "phẳng" bình thường hóa
        return cv2.normalize(h, h).flatten()
    
    #gọi histogram từ file
    def calcHistFromFile(file):
        img = cv2.imread(file)
        return calcHistogram(img)

    image = cv2.imread(f)
    # resize image trong khi vẫn giữ được khía cạnh ratio
    d = 1024 / image.shape[1]
    dim = (1024, int(image.shape[0] * d))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    # tạo một bản sao của hình ảnh để hiển thị kết quả
    output = image.copy()
    
    # convert image to grayscale
    #Chuyển đổi ảnh thành ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # cải thiện sự tương phản cho sự khác biệt trong điều kiện ánh sáng:
    #tạo một đối tượng CLAHE để áp dụng cân bằng biểu đồ thích ứng giới hạn tương phản
    #với -clipLimit=2.0: Ngưỡng giới hạn tương phản.
    #    -tileGridSize=(8, 8): Kích thước của lưới cho cân bằng biểu đồ.
    #                          Hình ảnh đầu vào sẽ được chia thành gạch hình chữ nhật 
    #                          có kích thước bằng nhau. brickGridSize xác định số lượng gạch 
    #                          trong hàng và cột.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # xác định lớp Enum ( Một lớp Enum có thể là:)
    class Enum(tuple): __getattr__ = tuple.index
    Material = Enum(('Copper', 'Brass', 'Euro1', 'Euro2'))
    # vị trí tập tin hình ảnh mẫu
    sample_images_copper = glob.glob("sample_images/copper/*")
    sample_images_brass = glob.glob("sample_images/brass/*")
    sample_images_euro1 = glob.glob("sample_images/euro1/*")
    sample_images_euro2 = glob.glob("sample_images/euro2/*")
    
    # Xác định dữ liệu huấn luyện và nhãn
    X = []
    y = []
    
    # Tính toán và lưu trữ dữ liệu và nhãn huấn luyện
    for i in sample_images_copper:
        X.append(calcHistFromFile(i))
        y.append(Material.Copper)
    for i in sample_images_brass:
        X.append(calcHistFromFile(i))
        y.append(Material.Brass)
    for i in sample_images_euro1:
        X.append(calcHistFromFile(i))
        y.append(Material.Euro1)
    for i in sample_images_euro2:
        X.append(calcHistFromFile(i))
        y.append(Material.Euro2)
    
    # instantiate classifier
    #Lớp khởi tạo
    # Nhiều lớp Perceptron
    # score: 0.974137931034
    #    MLP đào tạo trên hai mảng: mảng X có kích thước (n_samples, n_features),
    #   chứa các mẫu đào tạo được biểu diễn dưới dạng các vectơ đặc trưng của dấu phẩy động;
    #   và mảng y có kích thước (n_samples,), chứa các giá trị đích (nhãn lớp) cho các mẫu đào tạo:
        #Bộ giải để tối ưu hóa trọng lượng: solver="lbfgs" là một trình tối ưu hóa trong 
        #họ hàng của các phương pháp quasi-Newton.
    clf = MLPClassifier(solver="lbfgs") 
    
    # Phân chia các mẫu thành train và test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2)
    
    # Huấn luyện và điểm phân loại
    clf.fit(X_train, y_train)
    score = int(clf.score(X_test, y_test) * 100)
    print("độ chính xác TB: ", score)
    # Tạo label độ chính xác
    t2="Độ chính xác\nTB phân lớp:\n%d" %(score)
    label003 = Label(frame,text = "",font="Times 18 bold")
    label003.place(x=10,y=400)
    label003.config(text=t2)    
    
    #làm Mờ hình ảnh bằng cách sử dụng Gaussian blurring, nơi các điểm ảnh gần gũi hơn với trung tâm
    # Đóng góp nhiều hơn "weight" với mức trung bình, đối số đầu tiên là hình ảnh nguồn,
    #Đối số thứ hai là kernel size, thuws 3 lầ một sigma (0 for autodetect)
    #chúng ta dùng 7x7 kernel và Hãy để OpenCV Phát hiện sigma
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # circles: Một vectơ lưu trữ x, y, r cho mỗi vòng tròn được phát hiện.
    # src_gray: Input image (grayscale)
    # CV_HOUGH_GRADIENT: Xác định phương thức phát hiện.
    # dp = 2.2: Tỷ lệ nghịch của độ phân giải
    # min_dist = 100: Khoảng cách tối thiểu giữa các trung tâm được phát hiện
    # param_1 = 200: Ngưỡng trên cho máy dò cạnh Canny bên trong
    # param_2 = 100*: Ngưỡng phát hiện trung tâm
    # min_radius = 50: Bán kính tối thiểu được phát hiện.
    # max_radius = 120: Bán kính tối đa được phát hiện.
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=2.2, minDist=100,
                               param1=200, param2=100, minRadius=50, maxRadius=120)
     # todo:cấu trúc lại
    diameter = [] #mảng đường kính
    materials = [] #mảng chất liệu
    coordinates = [] #mảng tọa độ
    
    count = 0
    # Nếu phát hiện ít nhất 1 vòng tròn
    if circles is not None:
        #nối bán kính vào danh sách đường kính (không bận tâm nhân 2)
        for (x, y, r) in circles[0, :]:
            diameter.append(r)
    
        # chuyển đổi tọa độ và bán kính sang số nguyên
        circles = np.round(circles[0, :]).astype("int")
    
        # lặp qua tọa độ và bán kính của các vòng tròn
        for (x, y, d) in circles:
            count += 1    
            # thêm tọa độ vào danh sách
            coordinates.append((x, y))    
            # trích xuất vùng quan tâm
            roi = image[y - d:y + d, x - d:x + d]
    
            # thử công nhận loại vật liệu và thêm kết quả vào danh sách
            material = predictMaterial(roi)
            materials.append(material)
    
            # viết đồng tiền đeo mặt nạ vào tập tin            
            if False:
                m = np.zeros(roi.shape[:2], dtype="uint8")
                w = int(roi.shape[1] / 2)
                h = int(roi.shape[0] / 2)
                cv2.circle(m, (w, h), d, (255), -1)
                maskedCoin = cv2.bitwise_and(roi, roi, mask=m)
                cv2.imwrite("extracted/01coin{}.png".format(count), maskedCoin)
    
            #vẽ đường viền và kết quả trong hình ảnh đầu ra
            cv2.circle(output, (x, y), d, (0, 255, 0), 2)
            cv2.putText(output, material,
                        (x - 40, y), cv2.FONT_HERSHEY_PLAIN,
                        1.5, (0, 255, 0), thickness=2, 
                        lineType=cv2.LINE_AA)
    # lấy đường kính lớn nhất
    biggest = max(diameter)
    i = diameter.index(biggest)
    
    # quy mô mọi thứ theo đường kính tối đa
    # todo:cái này nên được chọn bởi người dùng
    if materials[i] == "Euro2":
        diameter = [x / biggest * 25.75 for x in diameter]
        scaledTo = "Scaled to 2 Euro"
    elif materials[i] == "Brass":
        diameter = [x / biggest * 24.25 for x in diameter]
        scaledTo = "Scaled to 50 Cent"
    elif materials[i] == "Euro1":
        diameter = [x / biggest * 23.25 for x in diameter]        
        scaledTo = "Scaled to 1 Euro"
    elif materials[i] == "Copper":
        diameter = [x / biggest * 21.25 for x in diameter]
        scaledTo = "Scaled to 5 Cent"
    else:
        scaledTo = "unable to scale.."
    
    i = 0
#    total = 0
    while i < len(diameter):
        d = diameter[i]
        m = materials[i]
        (x, y) = coordinates[i]
        t = "Unknown"
    
        # so sánh với đường kính đã biết với một số sai số
        if math.isclose(d, 25.75, abs_tol=1.25) and m == "Euro2":
            t = "2 Euro"
#            total += 200
        elif math.isclose(d, 23.25, abs_tol=2.5) and m == "Euro1":
            t = "1 Euro"
#            total += 100
        elif math.isclose(d, 19.75, abs_tol=1.25) and m == "Brass":
            t = "10 Cent"
#            total += 10
        elif math.isclose(d, 22.25, abs_tol=1.0) and m == "Brass":
            t = "20 Cent"
#            total += 20
        elif math.isclose(d, 24.25, abs_tol=2.5) and m == "Brass":
            t = "50 Cent"
#            total += 50
        elif math.isclose(d, 16.25, abs_tol=1.25) and m == "Copper":
            t = "1 Cent"
#            total += 1
        elif math.isclose(d, 18.75, abs_tol=1.25) and m == "Copper":
            t = "2 Cent"
#            total += 2
        elif math.isclose(d, 21.25, abs_tol=2.5) and m == "Copper":
            t = "5 Cent"
#            total += 5
    
        # write result on output image
        # viết kết quả hình output
        cv2.putText(output, t,
                    (x - 40, y + 22), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        i += 1
    
    # resize output image trong khi vẫn giữ được khía cạnh ratio
    d = 768 / output.shape[1]
    dim = (768, int(output.shape[0] * d))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    output = cv2.resize(output, dim, interpolation=cv2.INTER_AREA) 
    
    #chuyển anh numpy sang dạng h/anh
    arr2im = Image.fromarray(np.hstack([output])).resize((500,300))
    show_kq(arr2im)    
    def save1():
        arr2im.save('kq_save.png')
    button23 = Button(frame,width=12,text = "Save",fg = "#33CC00",
                     font="Times 14 bold",bg='Thistle2',command=save1)
    button23.place(x=50,y=350)

def showHuongdan():#button hướng dẫn
    root1 = Tk()
    
    root1.title("HƯỚNG DẪN SỬ DỤNG")
    frame1 = Frame(root1)
    frame1.pack(fill=BOTH, expand=1)
    
    str1="Chương trình được xây dựng bằng spyder (bản 3.6), để minh họa các phép toán như:\n"
    str2="Phép biến đổi hough cho đường thẳng, phép biến đổi hough cho hình tròn.\n\n"
    str3='-Khi bạn muốn  biến đổi đường thẳng bạn nhấn Nút Phát hiện đường thẳng\n'
    str4='-Và khi bạn muốn  biến đổi đường đường tròn bạn nhấn Nút Phát hiện đường tròn\n'
    str5='Khi bạn Nhấn 1 trong 2 nút trên thì nó sẽ xuất hiện 2 nút Load và xử lý \n'
    str6='Tiếp theo bạn Load hình ảnh từ thư mục sau đó nhấn nút Xử lý để xem kết quả\n'
   
    a = str1+str2+str3+str4+str5+str6
    label01 = Label(frame,fg = "black",font= "Times 20 bold")
    label01 = Label(frame1,text=a,
                    fg = "black",
                    font= "Times 14 bold").pack()
    label01.place(x=10,y=30)
    button9 = Button(frame1,
                     text = "Thoát",
                     command=root1.destroy,
                     fg = "black",
                     width=20,height=1)
    button9.place(x=30,y=250)

    root1.geometry("700x250")
    root1.mainloop()
    
def main():
    root = Tk()
    root.title("BTL MÔN NHẬP MÔN XỬ LÝ ẢNH-NHÓM 3-16DDS06031")
    global frame
    frame = Frame(root)
    frame.pack(fill=BOTH, expand=1)
    menubar = Menu(root)
    root.config(menu=menubar)

    fileMenu = Menu(menubar)
    menubar.add_cascade(label="File", menu=fileMenu)
    menubar.add_cascade(label="Help",command=showHuongdan)
    fileMenu.add_command(label="Close", command=root.destroy)
 
    gui1=win32gui.GetDesktopWindow()
    guicolor = win32gui.GetWindowDC(gui1) 
    
    #tên đề tài, GVHD, thành viên nhóm
    lbl = Label(root, text ="Môn học: NHẬP MÔN XỬ LÝ ẢNH",font = ("Times New Roman Bold",28),fg="red",bg='Moccasin')
    lbl.pack(expand=True)
    lbl.place(x=350,y=10)    
    label1 = Label(frame,text = "Đề Tài: Ứng dụng phát hiện đoạn thẳng và hình tròn bằng phép biến đổi Hough",
                   fg = "blue",bg='Moccasin',font = ("Times New Roman Bold",28))
    label1.place(x=50,y=50)
    label2 = Label(frame,text = "Giáo Viên Hướng Dẫn: Ngô Thanh Tú ",font="Times 20 bold")
    label2.place(x=160,y=100)
    label3 = Label(frame,text = "Thành viên nhóm 3: Đỗ Tống Quốc, Tôn Nữ Nguyên Hậu, Nguyễn Lê Xuân Phước ",font="Times 20 bold")
    label3.place(x=160,y=140)      
    
    # Lấy vị trí con trỏ set tọa độ & cường độ xám  
    def mousecoords(event):
        global lb2, lb4, guicolor
        #Tọa độ chuột
        x, y = event.x, event.y
        tc=('{}, {}'.format(x, y))
        lbl2.config(text=tc)
        
        #cường độ xám
        pixelcolor = win32gui.GetPixel(guicolor, event.x, event.y)
        b_color = int(pixelcolor)&0xff
        g_color = int(pixelcolor>>8)&0xff
        r_color = int(pixelcolor>>16)&0xff
        gray_level = int(0.3086*r_color + 0.6094*g_color + 0.0820*b_color)   
        lbl4.config(text=gray_level)
   
    lbl1 = Label(frame,text = "-Tọa độ chuột: ",font="Times 18 bold italic")
    lbl1.place(x=20,y=195)
    lbl2 = Label(frame,text = "0,0",font="Times 18 bold italic")
    lbl2.place(x=165,y=195)
    lbl3 = Label(frame,text = "-Cường độ xám: ",font="Times 18 bold italic")
    lbl3.place(x=300,y=195)
    lbl4 = Label(frame,text = "",font="Times 18 bold italic")
    lbl4.place(x=465,y=195)
    root.bind('<Motion>',mousecoords)  
    
    button0 = Button(frame,width=20,text = "Phát Hiện Đoạn thẳng",fg = "Dark Cyan",
                     font="Times 14 bold",bg='LightYellow2',command=executeline)
    button0.place(x=340,y=240)
    button1 = Button(frame,width=20,text = "Phát Hiện Đường Tròn",fg = "Dark Cyan",
                     font="Times 14 bold",bg='LightYellow2',command=circle_execute)
    button1.place(x=870,y=240)
    

    root.geometry("1350x680")
    root.mainloop()
    
if __name__=="__main__":
    main()
    
    
    
#def open_file():
#    filename = filedialog.askopenfilename(title="Select file", 
#                                              filetypes=(("MP4 files", "*.mp4"),
#                                              ('All files', '*')))
#    cap = cv2.VideoCapture(filename)
#    
#    ret, frame = cap.read()
#    cv2image   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#    img   = Image.fromarray(cv2image).resize((760, 400))    
#    imgtk = ImageTk.PhotoImage(image = img)
#    global lmain
#    lmain = Label(width=500,height=340)
#    lmain.imgtk = imgtk
#    lmain.configure(image=imgtk)
#    lmain.image=imgtk
#    lmain.place(x=200, y=290)
##    lmain.after(10, open_file)
#    cap.release()