
import cv2 as cv

if __name__ == '__main__':
    tt=cv.imread(r"ILSVRC2012_test_00000003.png",1)

    for i in range(tt.shape[0]):
        for j in range(tt.shape[1]):
               print(tt[i,j],end=" ")
        print()