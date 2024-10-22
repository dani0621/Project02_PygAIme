# Flappy Bird Game
# structure basically came from Tech with Tim YouTube video tutorials (any resource used is cited in my README)
# modified the inputs (added additional inputs) as well as wrote the program to visualize the neural networks
# modified the pipes so that they move up and down randomly to make it harder to train the agent
# also used ChatGPT to debug and install libraries (ran into trouble calling variables outside their scope)
# I don't know if I have to cite this, but I used autocorrect (for code) because PyCharm said the statement I used could be simplified

import pygame
import random
import os
import time
import neat
import visualize
import pickle
pygame.font.init()  # initialize the fonts used to display the text in the popup window

# setting up popup window parameters
WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# load images of bird flapping and pipe (only needed one the top/bottom is just turned 180 degrees)
pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())

gen = 0

# creates the bird agent
class Bird:
    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    # defaults the bird with how it would be without any movement
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0  # degrees to tilt
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    # makes the bird jump
    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    # sets up how the bird could move
    def move(self):
        self.tick_count += 1

        # for downward acceleration
        displacement = self.vel*(self.tick_count) + 0.5*(3)*(self.tick_count)**2  # calculate displacement

        # terminal velocity
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:  # tilt up (how much image tilts)
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            # tilt down
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    # draws the actual bird with images
    def draw(self, win):
        self.img_count += 1

        # for the bird to look like it's moving, it loops through three images
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        # so when bird is nose diving it isn't flapping, it's just going down
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2


        # tilt the bird
        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    # gets mask for current bird
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

# creates the pipes for the bird to navigate, originally used Tech With Tim, then modified so that the pipes would move up and down randomly
class Pipe():
    GAP = 200 # vertical gap between pipes
    VEL = 5 # speed at which the pipes move across the screen

    def __init__(self, x):
        # initializes the pipe and sets the height randomly (used Tech With Tim's initial code to help with this but modified the height range)
        self.x = x
        self.height = random.randrange(50, 450)
        self.top = self.height - pipe_img.get_height() * 2
        self.bottom = self.height + self.GAP # creates the position of the bottom pipe by using the top pipe's height and the gap previously defined
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True) # creates the top pipe by transforming the bottom pipe (instead of manually flipping this)
        self.PIPE_BOTTOM = pipe_img

        self.passed = False # indicates if bird had passed this pipe

        # Randomly sets vertical movement speed and direction (pipes aren't following a pattern)
        # Used ChatGPT for .uniform and .choice so that speed is a random floating number between 0.5 and 2 and then direction has to be a random selection of -1 or 1
        self.UP_DOWN_SPEED = random.uniform(0.5, 2)  # Random speed between 0.5 and 2
        self.direction = random.choice([-1, 1])  # random choice if it will move up or down

    # moves the pipe accordingly (adjusts height of the pipe in the right direction and speed)
    def move(self):
        self.x -= self.VEL

        # Move the pipe up or down
        self.height += self.direction * self.UP_DOWN_SPEED

        # changes direction if the pipe hits the limits of the border of popup, had help from ChatGPT because some would originally just completely disappear
        if self.height < 50 or self.height > 450:
            self.direction *= -1

        # updates the top and bottom positions
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    # draws the pipes at their positions
    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    # detects if the agent/bird has collided with a pipe (if the image of the bird and top/bottom pipe has overlapped in any place)
    def collide(self, bird, win):
        # Tech With Tim's code
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask,top_offset)

        if b_point or t_point:
            return True

        return False

# background/base of the game that makes the game look like it's moving
class Base:
    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img

    # initialized the floor
    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    # sets the movement so that it looks like it is moving by
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    # draws the actual floor so that the images move together
    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

# Tech With Tim
# rotates an image and then draw (blit) it onto a specified surface at a certain position
# makes sure that the image is centered at that position
def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)

    surf.blit(rotated_image, new_rect.topleft)

# creates the popup window for the main game loop
def draw_window(win, birds, pipes, base, score, gen, pipe_ind):
    if gen == 0:
        gen = 1
    win.blit(bg_img, (0,0))

    # draws pipes
    for pipe in pipes:
        pipe.draw(win)

    # draws base
    base.draw(win)

    # draw birds
    for bird in birds:
        # draw lines from bird to pipe
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
        # draw bird
        bird.draw(win)

    # Number of pipes passed/ Score
    score_label = STAT_FONT.render("Score: " + str(score),1,(255,255,255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    # which generation is being shown
    score_label = STAT_FONT.render("Gens: " + str(gen-1),1,(255,255,255))
    win.blit(score_label, (10, 10))

    # how many birds/agents are alive
    score_label = STAT_FONT.render("Alive: " + str(len(birds)),1,(255,255,255))
    win.blit(score_label, (10, 50))

    pygame.display.update()

# simulates the current generation birds and their fitness score (based on how far they get in the game)
def eval_genomes(genomes, config):
    global WIN, gen
    win = WIN
    gen += 1

    # Initializes lists to hold neural networks, bird objects, and genome references.
    # Each genome starts with a fitness of 0
    # A neural network is created for each genome, and a corresponding bird object is created at a fixed position (230, 350), same starting point
    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start each bird with fitness level of 0 in order to find the bird with the highest fitness score
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230,350))
        ge.append(genome)

    # game setup
    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()

    # defines FPS and game will run as long as it's active and there are birds alive
    run = True
    while run and len(birds) > 0:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():  # determines whether to use the first or second ipe on the screen for neural network input
                pipe_ind = 1

        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1 # give each bird a fitness reward of 0.1 for each frame it stays alive
            bird.move()

            # send bird location, top pipe location and bottom pipe location, bird's velocity, bird's distance from the pipe and distance frmo the bird to the ground
            # and determine from network whether the bird should jump or not
            output = nets[birds.index(bird)].activate((
                bird.y,  # Bird's vertical position
                abs(bird.y - pipes[pipe_ind].height),  # distance from the bird to the top of the pipe
                abs(bird.y - pipes[pipe_ind].bottom), # distance from bird to bottom of the pipe
                bird.vel,  # bird's velocity
                pipes[pipe_ind].x - bird.x, # horizontal distance from pipe to bird
                abs(FLOOR - bird.y) # added input of the distance from bird to the ground
            ))

            # If the output from the neural network suggests jumping (output > 0.5, meaning the probability is greater than 0.5), the bird jumps
            if output[0] > 0.5:
                bird.jump()

        # base and pipes update
        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            for bird in birds:
                if pipe.collide(bird, win): # checks if bird collided with pipe
                    ge[birds.index(bird)].fitness -= 1 #if bird did collide then the fitness score decreases
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1 # score increases with each additional pipe
            for genome in ge:
                genome.fitness += 5 # added reward for passing through a pipe
                if pipes[pipe_ind].height + 10 > bird.y > pipes[pipe_ind].bottom - 10: #ChatGPT helped with the indexing
                    ge[x].fitness += 2 # added reward if the bird flies close to the pipe without hitting it, allows the agent to slowly adjust to fitting inside the pipe
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        # Checks if birds go out of bounds and removes them if they do
        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        # updates window
        draw_window(WIN, birds, pipes, base, score, gen, pipe_ind)

        # limits the score
        if score > 200:
            pickle.dump(nets[0],open("best.pickle", "wb"))
            break


# runs the NEAT algorithm to train the neural network to play the game
def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population
    p = neat.Population(config)

    # reports the stats of each generation as they proceed
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 100 generations
    winner = p.run(eval_genomes, 100)

    # show final stats of the best agent
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')

    #ChatGPT help with writing config variable
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    winner = population.run(eval_genomes, 50)

    # plots how well the agents do over time (based on fitness score)
    visualize.plot_stats(stats, view=True)
    node_names = {0: 'Bird_Y', 1: 'Pipe_Top', 2: 'Pipe_Bottom', 3: 'Velocity', 4: 'Dist_Pipe', 5: 'Base_Dist',
                  -1: 'Flap'}
    visualize.draw_net(config, winner, view=True, node_names=node_names)

    run(config_path)
