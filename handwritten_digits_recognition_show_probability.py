import numpy as np
import pygame
from pygame import Vector2
from pygame import draw
from pygame.draw import circle, line, aaline, rect
from pygame.transform import scale, smoothscale
import pygame.display
import pygame.mouse
import pygame.event
import pygame.time
import neural_network
from handwritten_digits_recognition_tools import get_digit, image_to_input
import pygame.surfarray
import pygame.font

pygame.init()

GRAY = (50, 50, 50, 50)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WHITE_GRAY = (150, 150, 150, 50)
# weigth, heigth = 1080, 2160  # for phone
weigth,heigth=400,800#celiphone

screen = pygame.display.set_mode((weigth, heigth))
screen.fill(GRAY)
clock = pygame.time.Clock()

canvas_rect = pygame.Rect((0, heigth / 4), (weigth, weigth))
rect(screen, BLACK, canvas_rect)
digits_rect = pygame.Rect(((weigth / 2 - heigth / 8, 0), (heigth / 4, heigth / 4)))


# pen_size = 80
# font_size = 330
pen_size=15
font_size=150
digits_font = pygame.font.SysFont("", font_size)

input_layer = pygame.Surface(canvas_rect.size)
pixel_layer = pygame.Surface(canvas_rect.size, pygame.SRCALPHA)

last_pos = None


def offset(pos):
    """offset to canvas"""
    return Vector2(pos) - Vector2(canvas_rect.topleft)


def is_drawing(pos):
    return pygame.mouse.get_pressed(3)[0] and canvas_rect.collidepoint(pos)


network = neural_network.NeuralNetwork.create_from_file(
    "data/training_completed_data.json"
)

label_size = (30, 30)
label_leftop = Vector2(weigth / 22, heigth / 5)
label_offect = Vector2(weigth / 11, 0)
label = [
    (
        pygame.Surface(label_size),
        pygame.Rect(label_leftop + label_offect * n, label_size),
    )
    for n in range(10)
]


running = True
while running:
    clock.tick(30)

    pos = Vector2(pygame.mouse.get_pos())

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            last_pos = None
            if not canvas_rect.collidepoint(pos):
                input_layer.fill(BLACK)
                pixel_layer.fill(BLACK)

        if event.type == pygame.MOUSEBUTTONUP:
            last_pos = None

        if event.type == pygame.QUIT:
            running = False

    # drawing
    if is_drawing(pos):
        if last_pos == None:
            last_pos = pos

        line(input_layer, WHITE, offset(pos), offset(last_pos), pen_size)
        # circle(input_layer,WHITE,offset(pos),pen_size/2)
        # circle(input_layer,WHITE,offset((pos+last_pos)/2),pen_size/2)

        pixel_layer = smoothscale(input_layer, (28, 28))

        pixels_array = np.transpose(pygame.surfarray.pixels_red(pixel_layer)).reshape(
            784
        )

        digit = network.get(image_to_input(pixels_array))
        for n in range(len(label)):
            color = ((digit[n]) * 255, (digit[n]) * 255, (digit[n]) * 255)
            label[n][0].fill((color))
        digit_surface = digits_font.render(str(get_digit(digit)), True, WHITE)

        rect(screen, GRAY, digits_rect)

        for l, r in label:
            screen.blit(l, l.get_rect(center=r.center))

        screen.blit(digit_surface, digit_surface.get_rect(center=digits_rect.center))

    screen.blit(scale(pixel_layer, canvas_rect.size), canvas_rect)
    # screen.blit(input_layer,canvas_rect)
    pygame.display.update()

    last_pos = pos

pygame.quit()
