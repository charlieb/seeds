import numpy as np
from random import random
import svgwrite as svg
from math import pi, sin, cos
import numba as nb
from numba import jit, float64, int64, boolean
from itertools import product
import time
import concurrent.futures

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

ANG = 0 # Angle between successive seeds
SPD = 1 # Speed at which seeds move away from center per iteration
GRO = 2 # Growth rate of see per iteration
SSZ = 3 # Starting SiZe of the seed
CEN = 4 # Central clearance - middle can be clear
SEED_DEF_SIZE = 5 # MAX for ARRAY ITERATION and SIZING

# ----------- SEED GENERATION AND SCORING ---------------
@jit(boolean(float64[:], float64[:]), nopython=True, nogil=True)
def check_collision(seed1, seed2):
    return seed1[R] + seed2[R] > np.sqrt(np.sum((seed1[0:2] - seed2[0:2])**2))

@jit(float64(float64[:], float64[:]), nopython=True, nogil=True)
def dist(seed1, seed2):
    return np.sqrt(np.sum((seed1[0:2] - seed2[0:2])**2)) - (seed1[R] + seed2[R])

@jit((float64[:], float64[:], int64, int64), nopython=True, nogil=True)
def calc_seed(seed, seed_def, seed_number, iterations):
    my_iterations = iterations - seed_number # one seed per iteration
    seed[X] = cos(seed_def[ANG] * seed_number) * (seed_def[CEN] + seed_def[SPD] * my_iterations)
    seed[Y] = sin(seed_def[ANG] * seed_number) * (seed_def[CEN] + seed_def[SPD] * my_iterations)
    seed[R] = seed_def[SSZ] + seed_def[GRO] * my_iterations

@jit(float64(float64[:,:]), nopython=True, nogil=True)
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


@jit(float64(float64[:,:], float64[:]), nopython=True, nogil=True)
def generate_and_score(seeds, seed_def): # seeds for temporary storage
    for i in range(NSEEDS):
        calc_seed(seeds[i], seed_def, i, NSEEDS)
        for j in range(i):
            if dist(seeds[i], seeds[j]) < 0.0: # Hit!
                return 0

    # fitness function is the difference between the covered area and the
    # area of a circle with a radius that just encloses all the seeds
    return score_seeds(seeds)

@jit(nb.types.Tuple((int64, float64))(float64[:], int64), nopython=True, nogil=True)
def mt_gen_and_score(gene, i):
    seeds = np.zeros((NSEEDS, 3), dtype=np.float64)
    return i, generate_and_score(seeds, gene)

# ------------ ITERATIVE REFINEMENT --------------------------

@jit(float64[:]())
def plot_space():
    x,y,z = 1, 50, 50
    space = np.zeros((x,y,z), dtype=np.float64)
    seeds = np.zeros((NSEEDS, 3), dtype=np.float64)
    
    gene = np.zeros(SEED_DEF_SIZE, dtype=np.float64)
    # phi = 2.399963229
    gene[ANG] = 2.399963229 # 0.4668980 # random() * 2*pi
    gene[SSZ] = 0.0
    gene[CEN] = 0.0
    stepx, stepy, stepz = 2*pi, 4.0, 4.0
    bestx, besty, bestz = stepx/2., stepy/2., stepz/2.
    best = 0.
    for _ in range(4):
        rangex, rangey, rangez = stepx, stepy, stepz
        startx, starty, startz = bestx - stepx/2., besty - stepy/2., bestz - stepz/2.
        stepx, stepy, stepz = rangex/x, rangey/y, rangez/z

#        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#            gas = {executor.submit(mt_gen_and_score, pop[i], i) for i in range(pop_size)}
#            for future in concurrent.futures.as_completed(gas):
#                (i, score) = future.result()
#                scores[i] = score

        for i,j,k in product(range(x),range(y),range(z)): # numba can't cope with this in nopython mode
            #gene[ANG] = startx + i * stepx
            gene[GRO] = starty + j * stepy
            gene[SPD] = startz + k * stepz
            space[i][j][k] = generate_and_score(seeds, gene)
            if space[i][j][k] > best:
                best = space[i][j][k]
                bestx, besty, bestz = gene[ANG], gene[GRO], gene[SPD]
        print(best, bestx, besty, bestz)
    gene[ANG], gene[GRO], gene[SPD] = bestx, besty, bestz
    generate_and_score(seeds, gene)
    return seeds

# ------------- GENETIC SEARCH -----------------------------

#@jit((float64[:]))
def ransomize(gene):
    pass

@jit(float64(float64[:,:], int64))
def init_pop(pop, pop_size):
    for i in range(pop_size):
        pop[i][ANG] = 0.4668980 # random() * 2*pi
        pop[i][GRO] = random() * 2
        pop[i][SPD] = random() * 2


#@jit()#float64[:])
def ga():
    seeds = np.zeros((NSEEDS, 3), dtype=np.float64)
    pop_size = 1000
    pop = np.zeros((pop_size, SEED_DEF_SIZE), dtype=np.float64)
    next_pop = np.zeros((pop_size, SEED_DEF_SIZE), dtype=np.float64)
    init_pop(pop, pop_size)

    scores = np.zeros(pop_size, dtype=np.float64)

    best = -1.
    while True: # break happens in the generate_and_score section
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            gas = {executor.submit(mt_gen_and_score, pop[i], i) for i in range(pop_size)}
            for future in concurrent.futures.as_completed(gas):
                (i, score) = future.result()
                scores[i] = score

        best = -1.
        best_idx = -1
        for i in range(pop_size):
            best_idx = i if scores[i] > best else best_idx
            best = scores[i] if scores[i] > best else best
        if best > 0.4:
            generate_and_score(seeds, pop[best_idx])
            print(best, best_idx)
            return seeds
        print(best, pop[best_idx])

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


        # New blood
        for i in range(int(pop_size / 10)):
            next_pop[i][GRO] = random() * 2
            next_pop[i][SPD] = random() * 2
        # Mutation
        for i in range(int(pop_size / 10), 2*int(pop_size / 10)):
            #next_pop[i][ANG] += 0.01 * (0.5 - random())
            next_pop[i][GRO] += 0.1 * (0.5 - random())
            next_pop[i][SPD] += 0.1 * (0.5 - random())

        # Keep the best
        next_pop[0] = pop[best_idx]

        pop, next_pop = next_pop, pop
            
    return seeds

# ------------------ HILL CLIMBING ------------------------
#
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
            calc_seed(seeds[i], pos, i, iterations)
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

# -------------------- DRAWING AND MAIN -----------------
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
    seeds = plot_space()
    #seeds = add_seeds3(NSEEDS)
    #seeds = ga()
    dwg = svg.Drawing('test.svg')
    dwg.set_desc(title='Seeds', desc='My seed packet')
    draw(seeds, dwg)
    dwg.save()


if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()

    print("Elapsed:", t1-t0)
