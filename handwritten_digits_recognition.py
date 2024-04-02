from random import random
import numpy as np
import pygame
from pygame import Vector2
from pygame import draw
from pygame import surface
from pygame import transform
from pygame.draw import circle,line,aaline,rect
from pygame.transform import scale,smoothscale
import pygame.display
import pygame.mouse
import pygame.event
import pygame.time
import neural_network
from handwritten_digits_recognition_tools import get_digit,image_to_input
import pygame.surfarray
import pygame.font
import pygame.transform
pygame.init()

GRAY=(50,50,50,50)
BLACK=(0,0,0)
WHITE=(255,255,255)
WHITE_GRAY=(150,150,150,50)
#weigth,heigth=1080,2160 #for phone
weigth,heigth=400,800#celiphone

screen=pygame.display.set_mode((weigth,heigth))
screen.fill(GRAY)
clock=pygame.time.Clock()

canvas_rect=pygame.Rect((0,heigth/4),(weigth,weigth))
rect(screen,BLACK,canvas_rect)
digits_rect=pygame.Rect(((weigth/2-heigth/8,0),(heigth/4,heigth/4)))


pen_size=15
font_size=150
digits_font= pygame.font.SysFont("",font_size)

input_layer=pygame.Surface(canvas_rect.size)
pixel_layer=pygame.Surface(canvas_rect.size,pygame.SRCALPHA)

last_pos=None


def offset(pos):
    '''offset to canvas'''
    return Vector2(pos)-Vector2(canvas_rect.topleft)

def is_drawing(pos):
    return pygame.mouse.get_pressed(3)[0] and canvas_rect.collidepoint(pos)

path="data/tcd_best.json"
network=neural_network.NeuralNetwork.create_from_file(path)

rotate_range=2
zoom_range=0.02
move_range=0.1
offset_num=10
running=True
while running:
    clock.tick(30)

    pos=Vector2(pygame.mouse.get_pos())

    for event in pygame.event.get():
        if event.type==pygame.MOUSEBUTTONDOWN:
            last_pos=None
            if not canvas_rect.collidepoint(pos):
                input_layer.fill(BLACK)
                pixel_layer.fill(BLACK)

        if event.type == pygame.MOUSEBUTTONUP:
            last_pos=None

        if event.type==pygame.QUIT:
            running=False


    #drawing
    if is_drawing(pos):
        if last_pos==None:
            last_pos=pos
            
        line(input_layer,WHITE,offset(pos),offset(last_pos),pen_size)
        pixel_layer=smoothscale(input_layer,(28,28))
        pixels_array= np.transpose(pygame.surfarray.pixels_red(pixel_layer)).reshape(784)
        
        
        '''average_digit=np.zeros((10,1))
        for _ in range(offset_num):
            offset_pixel_layer=pygame.Surface((28,28),pygame.SRCALPHA)
            offset_pixel_layer.fill((0,0,0,0))
            offset_pixel_layer.blit(pygame.transform.rotozoom(pixel_layer,
               np.random.randn()*rotate_range,1+np.random.randn()*zoom_range),
                (np.random.randn()*move_range,np.random.randn()*move_range))
            screen.blit(scale(offset_pixel_layer,canvas_rect.size),canvas_rect)

            pixels_array= np.transpose(pygame.surfarray.pixels_red(offset_pixel_layer)).reshape(784)
            average_digit+=network.get(image_to_input(pixels_array))
        digit=average_digit/offset_num
        '''
        digit=network.get(image_to_input(pixels_array))
        digit_surface=digits_font.render(str(get_digit(digit)),True,WHITE)

        rect(screen,GRAY,digits_rect)
        screen.blit(digit_surface,digit_surface.get_rect(center= digits_rect.center))  
        
    screen.blit(scale(pixel_layer,canvas_rect.size),canvas_rect)
    #screen.blit(input_layer,canvas_rect)
    pygame.display.update()

    last_pos=pos
    
pygame.quit()