import numpy as np
from random import random
import svgwrite as svg
from math import pi, sin, cos
from numba import jit, float64, int64, boolean
from itertools import product

X = 0
Y = 1
R = 2
F = 3 # Fixed
VX = 4
VY = 5

NSEEDS = 200
ITERATE_STEP = 0.05
ZERO_POS_SCALE = 0.0
R_SEED = 0.5
ERROR = 0.01

STOP_R = 10.0

ANG = 0
SPD = 1
GRO = 2

@jit(boolean(float64[:], float64[:]))
def check_collision(seed1, seed2):
    return seed1[R] + seed2[R] > np.sqrt(np.sum((seed1[0:2] - seed2[0:2])**2))

@jit(float64(float64[:], float64[:]))
def dist(seed1, seed2):
    return np.sqrt(np.sum((seed1[0:2] - seed2[0:2])**2)) - (seed1[R] + seed2[R])

@jit((float64[:], float64, int64 ,float64, float64, int64))
def calc_seed(seed, angle, seed_number, velocity, growth_rate, iterations):
    my_iterations = iterations - seed_number # one seed per iteration
    seed[X] = cos(angle * seed_number) * velocity * my_iterations
    seed[Y] = sin(angle * seed_number) * velocity * my_iterations
    seed[R] = growth_rate * my_iterations


@jit(float64(float64[:,:]))
def score_seeds(seeds):
    max_r = -1.
    area = 0.
    for i in range(NSEEDS):
        r = np.sqrt(np.sum(seeds[i][0:2]**2)) + seeds[i][R]
        max_r = r if r > max_r else max_r

        area += pi * seeds[i][R]**2
    if max_r > 0.0:
        return area / (pi * max_r**2)
    else:
        return 0


@jit(float64(float64[:,:], float64[:]))
def generate_and_score(seeds, pos): # seeds for temporary storage
    for i in range(NSEEDS):
        calc_seed(seeds[i], pos[ANG], i, pos[SPD], pos[GRO], NSEEDS * 1.0) #####  !!!!!! #####
        for j in range(i):
            if dist(seeds[i], seeds[j]) < 0.0: # Hit!
                return 0

    # fitness function is the difference between the covered area and the
    # area of a circle with a radius that just encloses all the seeds
    return score_seeds(seeds)

#@jit((float64[:]))
def ransomize(gene):
    gene[ANG] = 0.4668980 # random() * 2*pi
    gene[GRO] = random() * 2
    gene[SPD] = random() * 2

@jit(float64(float64[:,:], int64))
def init_pop(pop, pop_size):
    for i in range(pop_size):
        ransomize(pop[i])

def plot_space():
    x,y = 20, 20
    space = np.zeros((x,y), dtype=np.float64)
    seeds = np.zeros((NSEEDS, 3), dtype=np.float64)
    
    gene = np.zeros(3, dtype=np.float64)
    gene[ANG] = 0.4668980 # random() * 2*pi
    for i,j in product(range(x),range(y)):
        gene[GRO] = i / x
        gene[SPD] = j / y
        space[i][j] = generate_and_score(seeds, gene)
    print(space)


#@jit(nopython=True)
def ga():
    seeds = np.zeros((NSEEDS, 3), dtype=np.float64)
    pop_size = 1000
    pop = np.zeros((pop_size, 3), dtype=np.float64)
    next_pop = np.zeros((pop_size, 3), dtype=np.float64)
    init_pop(pop, pop_size)

    scores = np.zeros(pop_size, dtype=np.float64)

    while True:
        for i in range(pop_size):
            scores[i] = generate_and_score(seeds, pop[i])

        best = -1.
        best_idx = -1
        for i in range(pop_size):
            best_idx = i if scores[i] > best else best_idx
            best = scores[i] if scores[i] > best else best
        if best > 0.8:
            generate_and_score(seeds, pop[best_idx])
            print(best, best_idx)
            return seeds
        print(best)

        total = 0.
        for i in range(pop_size):
            total += scores[i]
            scores[i] = total

        for i in range(pop_size):
            val = random() * total
            sel1 = 0
            while scores[sel1] < val:
                sel1 += 1

            val = random() * total
            sel2 = 0
            while scores[sel2] < val:
                sel2 += 1

            next_pop[i] = (pop[sel1] + pop[sel2]) / 2.

        tmp = pop
        pop = next_pop
        next_pop = tmp

        for i in range(int(pop_size / 10)):
            ransomize(pop[i])
        # Keep the best
        pop[0] = next_pop[best_idx]
            
    print(scores)
    return seeds

#@jit(float64[:,:]())
def gen_head():
    seeds = np.zeros((NSEEDS, 3))
    growth = 0.01
    iterations = NSEEDS * 1.05
    #angle = 123.1 * pi/180.0
    #angle = random() * pi #/ 16
    angle = 0.4668980
    v = 0.5# * random() #R_SEED*2 / (2*pi / angle_d)

    pos = np.array([angle, v, growth], dtype=np.float64)
    vec = np.array([angle, v, growth], dtype=np.float64)
    vec[ANG] = 0.001 * (0.5 - random()) # Angle
    vec[SPD] = 0.001 * (0.5 - random()) # Vector
    vec[GRO] = 0.001 * (0.5 - random()) # Growth

    # Must have a starting position with no hits!
    success = False
    hit = True
    score = 0
    last_score = -1.
    its = -1
    while score < 0.7 and its < 2000:
        its += 1
        if score > last_score or its % 100 == 0:
            #print(score, last_score, hit, pos[1:], vec[1:])
            print(its, score, last_score)
        # Apply offset
        pos += vec
        # Position all the seeds but maybe generate a hit
        for i in range(NSEEDS):
            calc_seed(seeds[i], pos[ANG], i, pos[SPD], pos[GRO], iterations)
            hit = False
            for j in range(i):
                if dist(seeds[i], seeds[j]) < 0.0:
                    hit = True
                    break # goto success loop (outermost)
            if hit: break

        # fitness function is the difference between the covered area and the
        # area of a circle with a radius that just encloses all the seeds
        if not hit:
            score = score_seeds(seeds)

        # Need a new direction?
        if hit or last_score > score:
            # go back first then pick a new direction
            pos -= vec
            #print(score, last_score, score_seeds(seeds))
            while True: # Make sure we're not creating invisible circles
                vec[ANG] = 0.001 * (0.5 - random()) # Angle
                vec[SPD] = 0.001 * (0.5 - random()) # Vector
                vec[GRO] = 0.001 * (0.5 - random()) # Growth
                tst = pos + vec
                if tst[GRO] > 0. and tst[SPD] > 0.:
                    break
        else: # not hit and lesser distance
            last_score = score

    print(score)
    return seeds

@jit(float64(float64[:], float64))
def radius_error(seed, radius):
    return radius - np.sqrt(np.sum(seed[0:2]**2))

@jit(float64(float64[:], float64[:]))
def collision_error(seed1, seed2):
    return np.sqrt(np.sum((seed1[0:2] - seed2[0:2])**2)) - (seed1[R] + seed2[R])

@jit((float64[:], float64[:], float64, int64))
def minimize_error(seed, seed2, radius, err_selector):
    scale = 1.0
    error = radius_error(seed, radius) if err_selector == R else collision_error(seed, seed2)
    while abs(error) > ERROR:
        scale *= -0.5
        prev_error = error * 2
        while abs(prev_error) > abs(error) and abs(error) > ERROR:
            seed[0:2] += seed[VX:VX+2] * scale
            prev_error = error
            error = radius_error(seed, radius) if err_selector == R else collision_error(seed, seed2)

@jit(float64(float64[:,:], int64, float64))
def iterate3(seeds, nseeds, max_radius):
    acc = np.array([0,0], dtype=np.float64)

    for i1 in range(nseeds):
        seeds[i1][R] += 0.01
        if seeds[i1][F] != 0.0:
            seeds[i1][R] += 0.01
            continue
        seeds[i1][0:2] += seeds[i1][VX:VX+2]

        #for i2 in range(nseeds):
        #    if i1 == i2: continue
        #    # don't collide with moving seeds
        #    if seeds[i2][F] == 0.0: continue

        #    delta = seeds[i1][0:2] - seeds[i2][0:2]
        #    d = np.sqrt(np.sum(delta**2))
        #    # Collision with fixed seed -> fix
        #    if d < seeds[i1][R] + seeds[i2][R]:
        #        seeds[i1][F] = 1.0
        #        minimize_error(seeds[i1], seeds[i2], max_radius, 0)
        #        break

        # no more motion for fixed seeds
        #if seeds[i1][F] != 0.0: continue

        #if max_radius <= np.sqrt(np.sum(seeds[i1][0:2]**2)):
        #    seeds[i1][F] = 1.0
        #    minimize_error(seeds[i1], seeds[i1], max_radius, R)
        
    return max_radius

@jit(float64[:,:](int64))
def add_seeds3(nseeds):
    seeds = np.zeros((nseeds, 6))
    angle = 0.0
    #angle_d = 123.1 * pi/180.0
    #angle_d = random() * pi #/ 16
    angle_d = 0.4668980
    v = 0.2 * random() #R_SEED*2 / (2*pi / angle_d)

    print(angle_d, v)
    for i in range(nseeds):
        angle += angle_d
        seeds[i][X] = cos(angle) * ZERO_POS_SCALE
        seeds[i][Y] = sin(angle) * ZERO_POS_SCALE
        seeds[i][R] = R_SEED
        seeds[i][F] = 0.0
        seeds[i][VX] = cos(angle) * v
        seeds[i][VY] = sin(angle) * v
        iterate3(seeds, i+1, STOP_R)
    return seeds

np.seterr(all='raise')
@jit(float64(float64[:,:], int64, float64))
def iterate2(seeds, nseeds, radius):
    cent = np.array([0,0], dtype=np.float64)

    cent[X] = cent[Y] = 0.0
    for i in range(nseeds):
        cent += seeds[i][0:2]
    cent /= nseeds

    for i in range(nseeds):
        delta = seeds[i][0:2] - cent
        d = np.sqrt(np.sum(delta**2))
        if d != 0.0:
            seeds[i][0:2] += ITERATE_STEP * delta / d

    return radius

@jit(float64[:,:](int64))
def add_seeds2(nseeds):
    seeds = np.zeros((nseeds, 3))
    angle = random() * 2*pi
    seeds[0][X] = 0 #cos(angle) * R_SEED
    seeds[0][Y] = 0.5 #sin(angle) * R_SEED
    seeds[0][R] = R_SEED
    radius = 2*R_SEED

    for i in range(1,nseeds):
        angle = random() * 2*pi
        seeds[i][X] = cos(angle) * ZERO_POS_SCALE
        seeds[i][Y] = sin(angle) * ZERO_POS_SCALE
        seeds[i][R] = R_SEED
        #while radius > np.sqrt(np.sum(seeds[i][0:2]**2)):
        for _ in range(10):
            radius = iterate2(seeds, i+1, radius)
    return seeds

@jit(float64(float64[:,:], int64, float64))
def iterate(seeds, nseeds, max_radius):
    acc = np.array([0,0], dtype=np.float64)

    for i1 in range(nseeds):
        if seeds[i1][F] != 0.0: continue

        acc[X] = acc[Y] = 0.0
        for i2 in range(nseeds):
            if i1 == i2: continue
            delta = seeds[i1][0:2] - seeds[i2][0:2]
            d = np.sqrt(np.sum(delta**2))
            # Collision with fixed seed -> fix
            if seeds[i2][F] != 0.0 and d < seeds[i1][R] + seeds[i2][R]:
                seeds[i1][F] = 1.0

                scale = 1.0
                error = d - (seeds[i1][R] + seeds[i2][R])
                prev_error = error * 2
                while abs(error) > ERROR:
                    scale *= -0.5
                    prev_error = error * 2
                    while abs(prev_error) > abs(error):
                        seeds[i1][0:2] += seeds[i1][VX:VX+2] * scale

                        delta = seeds[i1][0:2] - seeds[i2][0:2]
                        d = np.sqrt(np.sum(delta**2))
                        prev_error = error
                        error = d - (seeds[i1][R] + seeds[i2][R])

                # Adust the radius
                max_radius = np.sqrt(np.sum(seeds[i1][0:2]**2))
                break

            acc += delta / d**2

        if acc[X] + acc[Y] != 0.0:
            acc /= np.sqrt(np.sum(acc**2))
                
            seeds[i1][VX:VX+2] = ITERATE_STEP * acc
            seeds[i1][0:2] += seeds[i1][VX:VX+2]

        if max_radius <= np.sqrt(np.sum(seeds[i1][0:2]**2)):
            seeds[i1][F] = 1.0
        
        return max_radius

@jit(float64[:,:](int64))
def add_seeds(nseeds):
    seeds = np.zeros((nseeds, 6))
    angle = random() * 2*pi
    seeds[0][X] = cos(angle) * R_SEED * 2.1
    seeds[0][Y] = sin(angle) * R_SEED * 2.1
    seeds[0][R] = R_SEED
    seeds[0][F] = 0.0
    radius = STOP_R
    for i in range(1,nseeds):
        print(i)
        angle = random() * 2*pi
        seeds[i][X] = cos(angle) * ZERO_POS_SCALE
        seeds[i][Y] = sin(angle) * ZERO_POS_SCALE
        seeds[i][R] = R_SEED
        seeds[i][F] = 0.0
        # Make sure the seed is clear of the starting area
        while R_SEED + ZERO_POS_SCALE >= np.sqrt(np.sum(seeds[i][0:2]**2)):
            if seeds[i][F] > 0.0:
                break
            radius = iterate(seeds, i+1, radius)
    return seeds

def draw(seeds, drawing):
    minx = miny = 999999
    maxx = maxy = -999999
    for s in seeds:
        if s[X] - s[R] < minx: minx = s[X] - s[R]
        if s[Y] - s[R] < miny: miny = s[Y] - s[R]
        if s[X] + s[R] > maxx: maxx = s[X] + s[R]
        if s[Y] + s[R] > maxy: maxy = s[Y] + s[R]
        c = svg.shapes.Circle((s[X], s[Y]), s[R],
                                    #fill='none', 
                                    fill='white', 
                                    stroke='black',
                                    stroke_width=1.0)
        drawing.add(c)
    drawing.viewbox(minx=minx, miny=miny, 
                    width=maxx-minx, height=maxy-miny)

def main():
    plot_space()
    return
    #seeds = add_seeds3(NSEEDS)
    seeds = ga()
    dwg = svg.Drawing('test.svg')
    dwg.set_desc(title='Seeds', desc='My seed packet')
    draw(seeds, dwg)
    dwg.save()


if __name__ == '__main__':
    main()
