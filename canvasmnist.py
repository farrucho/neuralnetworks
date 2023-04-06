import pygame
import random
import numpy as np
import cv2
import PIL
from PIL import Image

# Making canvas
screen = pygame.display.set_mode((200, 200))
 
# Setting Title
pygame.display.set_caption('Canvas')
 
 
draw_on = False
last_pos = (0, 0)
 
# Radius of the Brush
radius = 3
 
 
def roundline(canvas, color, start, end, radius=1):
    Xaxis = end[0]-start[0]
    Yaxis = end[1]-start[1]
    dist = max(abs(Xaxis), abs(Yaxis))
    for i in range(dist):
        x = int(start[0]+float(i)/dist*Xaxis)
        y = int(start[1]+float(i)/dist*Yaxis)
        pygame.draw.circle(canvas, color, (x, y), radius)
 
 
try:
    while True:
        e = pygame.event.wait()
         
        if e.type == pygame.QUIT:
            raise StopIteration
             
        if e.type == pygame.MOUSEBUTTONDOWN:
            color = (255,255,255)
            if e.button == 1:
                color = (255,255,255)
            else:
                color = (0,0,0)
            # Draw a single circle wheneven mouse is clicked down.
            pygame.draw.circle(screen, color, e.pos, radius)
            draw_on = True
        # When mouse button released it will stop drawing   
        if e.type == pygame.MOUSEBUTTONUP:
            draw_on = False
        # It will draw a continuous circle with the help of roundline function.   
        if e.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.draw.circle(screen, color, e.pos, radius)
                roundline(screen, color, e.pos, last_pos,  radius)
            last_pos = e.pos

        
        pygame.image.save(screen, "canvas.jpeg")
    
        size = (28,28)
        img = cv2.imread("canvas.jpeg")
        img = cv2.resize(img,size,interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("canvas.jpeg", img)

        array = np.array(img).astype('float32')/255

        array = np.mean(array, axis=2)
        array = array.reshape(1,1,28*28)

        from mnist import *
        out = np.array(net.predict(array))
        out = out.squeeze().reshape(10,1)

        print(f'probabilidades: 0:{round(float(out[0][0]),2)} 1:{round(float(out[1][0]),2)} 2:{round(float(out[2][0]),2)} 3:{round(float(out[3][0]),2)} 4:{round(float(out[4]),2)} 5:{round(float(out[5][0]),2)} 6:{round(float(out[6][0]),2)} 7:{round(float(out[7][0]),2)} 8:{round(float(out[8][0]),2)} 9:{round(float(out[9][0]),2)}\n VALOR MAIS PROVAVEL: {np.argmax(out)}, prob: {np.max(out)*100}%')

        pygame.display.flip()
 
except StopIteration:
    pygame.quit()


"""     print(array)
    array = array.reshape(1, 28*28)
    from mnist import *
    print(net.predict(array)) """

# Quit
pygame.quit()