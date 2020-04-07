from projet import  *
# Import and initialize the pygame library
import pygame
pygame.init()
screen = pygame.display.set_mode([500, 500])

def drawGrid(screen):
    blockSize = 20 #Set the size of the grid block
    for x in range(WINDOW_WIDTH):
        for y in range(WINDOW_HEIGHT):
            rect = pygame.Rect(x*blockSize, y*blockSize,
                               blockSize, blockSize)
            pygame.draw.rect(screen, WHITE, rect, 1)

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
WINDOW_HEIGHT = 400
WINDOW_WIDTH = 400
width = 10
length = 10
rl = RL(0.9, 0.7, 10, 10)

rl.add_hunter(1, 1)
rl.add_prey(9, 9)

# Run until the user asks to quit
running = True
while running:
    rl.episode()
    #drawGrid(screen)
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Fill the background with white
    screen.fill((0, 0, 0))
    agents = rl.agents
    # Draw a solid blue circle in the center
    for i in agents:
        if i.type == "hunter":
            pygame.draw.rect(screen, (0, 0, 255), (i.posx, i.posy, 10, 10))
        if i.type == "prey":
            pygame.draw.circle(screen, (0, 0, 255), (i.posx, i.posy), 10)
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()