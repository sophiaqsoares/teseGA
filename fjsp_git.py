#!/usr/bin/env python



import sys
import time
import random
import itertools
import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import colors as mcolors


# Config
popSize = 40   #400
maxGen = 10    #200
pr = 0.005
pc = 0.8
pm = 0.1
latex_export = False

path = "/Users/sophi/Downloads/GA_Sophia/data/test.fjs"


colors = []
for name, hex in mcolors.cnames.items():
    colors.append(name)

# Parser
def parse(path):
    file = open(path, 'r')

    firstLine = file.readline()
    firstLineValues = list(map(int, firstLine.split()[0:2]))

    jobsNb = firstLineValues[0]
    machinesNb = firstLineValues[1]

    jobs = []

    for i in range(jobsNb):
        currentLine = file.readline()
        currentLineValues = list(map(int, currentLine.split()))

        operations = []

        j = 1
        while j < len(currentLineValues):
            k = currentLineValues[j]
            j = j+1

            operation = []

            for ik in range(k):
                machine = currentLineValues[j]
                j = j+1
                processingTime = currentLineValues[j]
                j = j+1

                operation.append({'machine': machine, 'processingTime': processingTime})

            operations.append(operation)

        jobs.append(operations)

    file.close()

    return {'machinesNb': machinesNb, 'jobs': jobs}


# Encoding
def generateOS(parameters):
    jobs = parameters['jobs']

    OS = []
    i = 0
    for job in jobs:
        for op in job:
            OS.append(i)
        i = i+1

    random.shuffle(OS)

    return OS

def generateMS(parameters):
    jobs = parameters['jobs']

    MS = []
    for job in jobs:
        for op in job:
            randomMachine = random.randint(0, len(op)-1)
            MS.append(randomMachine)

    return MS

def initializePopulation(parameters):
    gen1 = []

    for i in range(popSize):
        OS = generateOS(parameters)
        MS = generateMS(parameters)
        gen1.append((OS, MS))

    return gen1


# Decoding
def split_ms(pb_instance, ms):
    jobs = []
    current = 0
    for index, job in enumerate(pb_instance['jobs']):
        jobs.append(ms[current:current+len(job)])
        current += len(job)
    return jobs


def get_processing_time(op_by_machine, machine_nb):
    for op in op_by_machine:
        if op['machine'] == machine_nb:
            return op['processingTime']
    print("[ERROR] Machine {} doesn't to be able to process this task.".format(machine_nb))
    sys.exit(-1)


def is_free(tab, start, duration):
    for k in range(start, start+duration):
        if not tab[k]:
            return False
    return True


def find_first_available_place(start_ctr, duration, machine_jobs):
    max_duration_list = []
    max_duration = start_ctr + duration

    # max_duration is either the start_ctr + duration or the max(possible starts) + duration
    if machine_jobs:
        for job in machine_jobs:
            max_duration_list.append(job[3] + job[1])  # start + process time

        max_duration = max(max(max_duration_list), start_ctr) + duration

    machine_used = [True] * max_duration

    # Updating array with used places
    for job in machine_jobs:
        start = job[3]
        long = job[1]
        for k in range(start, start + long):
            machine_used[k] = False

    # Find the first available place that meets constraint
    for k in range(start_ctr, len(machine_used)):
        if is_free(machine_used, k, duration):
            return k


def decode(pb_instance, os, ms):
    o = pb_instance['jobs']
    machine_operations = [[] for i in range(pb_instance['machinesNb'])]

    ms_s = split_ms(pb_instance, ms)  # machine for each operations

    indexes = [0] * len(ms_s)
    start_task_cstr = [0] * len(ms_s)

    # Iterating over OS to get task execution order and then checking in
    # MS to get the machine
    for job in os:
        index_machine = ms_s[job][indexes[job]]
        machine = o[job][indexes[job]][index_machine]['machine']
        prcTime = o[job][indexes[job]][index_machine]['processingTime']
        start_cstr = start_task_cstr[job]

        # Getting the first available place for the operation
        start = find_first_available_place(start_cstr, prcTime, machine_operations[machine - 1])
        name_task = "{}-{}".format(job, indexes[job]+1)

        machine_operations[machine - 1].append((name_task, prcTime, start_cstr, start))

        # Updating indexes (one for the current task for each job, one for the start constraint
        # for each job)
        indexes[job] += 1
        start_task_cstr[job] = (start + prcTime)

    return machine_operations


def translate_decoded_to_gantt(machine_operations):
    data = {}

    for idx, machine in enumerate(machine_operations):
        machine_name = "Machine-{}".format(idx + 1)
        operations = []
        for operation in machine:
            operations.append([operation[3], operation[3] + operation[1], operation[0]])

        data[machine_name] = operations

    return data

# Termination
def shouldTerminate(population, gen):
    return gen > maxGen

# Genetic

def timeTaken(os_ms, pb_instance):
    (os, ms) = os_ms
    decoded = decode(pb_instance, os, ms)

    # Getting the max for each machine
    max_per_machine = []
    for machine in decoded:
        max_d = 0
        for job in machine:
            end = job[3] + job[1]
            if end > max_d:
                max_d = end
        max_per_machine.append(max_d)
    return max(max_per_machine)


# 4.3.1 Selection
#######################

def elitistSelection(population, parameters):
    keptPopSize = int(pr * len(population))
    sortedPop = sorted(population, key=lambda cpl: timeTaken(cpl, parameters))
    return sortedPop[:keptPopSize]


def tournamentSelection(population, parameters):
    b = 2

    selectedIndividuals = []
    for i in range(b):
        randomIndividual = random.randint(0, len(population) - 1)
        selectedIndividuals.append(population[randomIndividual])

    return min(selectedIndividuals, key=lambda cpl: timeTaken(cpl, parameters))


def selection(population, parameters):
    newPop = elitistSelection(population, parameters)
    while len(newPop) < len(population):
        newPop.append(tournamentSelection(population, parameters))

    return newPop


# 4.3.2 Crossover operators
###########################

def precedenceOperationCrossover(p1, p2, parameters):
    J = parameters['jobs']
    jobNumber = len(J)
    jobsRange = range(1, jobNumber+1)
    sizeJobset1 = random.randint(0, jobNumber)

    jobset1 = random.sample(jobsRange, sizeJobset1)

    o1 = []
    p1kept = []
    for i in range(len(p1)):
        e = p1[i]
        if e in jobset1:
            o1.append(e)
        else:
            o1.append(-1)
            p1kept.append(e)

    o2 = []
    p2kept = []
    for i in range(len(p2)):
        e = p2[i]
        if e in jobset1:
            o2.append(e)
        else:
            o2.append(-1)
            p2kept.append(e)

    for i in range(len(o1)):
        if o1[i] == -1:
            o1[i] = p2kept.pop(0)

    for i in range(len(o2)):
        if o2[i] == -1:
            o2[i] = p1kept.pop(0)

    return (o1, o2)


def jobBasedCrossover(p1, p2, parameters):
    J = parameters['jobs']
    jobNumber = len(J)
    jobsRange = range(0, jobNumber)
    sizeJobset1 = random.randint(0, jobNumber)

    jobset1 = random.sample(jobsRange, sizeJobset1)
    jobset2 = [item for item in jobsRange if item not in jobset1]

    o1 = []
    p1kept = []
    for i in range(len(p1)):
        e = p1[i]
        if e in jobset1:
            o1.append(e)
            p1kept.append(e)
        else:
            o1.append(-1)

    o2 = []
    p2kept = []
    for i in range(len(p2)):
        e = p2[i]
        if e in jobset2:
            o2.append(e)
            p2kept.append(e)
        else:
            o2.append(-1)

    for i in range(len(o1)):
        if o1[i] == -1:
            o1[i] = p2kept.pop(0)

    for i in range(len(o2)):
        if o2[i] == -1:
            o2[i] = p1kept.pop(0)

    return (o1, o2)


def twoPointCrossover(p1, p2):
    pos1 = random.randint(0, len(p1) - 1)
    pos2 = random.randint(0, len(p1) - 1)

    if pos1 > pos2:
        pos2, pos1 = pos1, pos2

    offspring1 = p1
    if pos1 != pos2:
        offspring1 = p1[:pos1] + p2[pos1:pos2] + p1[pos2:]

    offspring2 = p2
    if pos1 != pos2:
        offspring2 = p2[:pos1] + p1[pos1:pos2] + p2[pos2:]

    return (offspring1, offspring2)


def crossoverOS(p1, p2, parameters):
    if random.choice([True, False]):
        return precedenceOperationCrossover(p1, p2, parameters)
    else:
        return jobBasedCrossover(p1, p2, parameters)


def crossoverMS(p1, p2):
    return twoPointCrossover(p1, p2)


def crossover(population, parameters):
    newPop = []
    i = 0
    while i < len(population):
        (OS1, MS1) = population[i]
        (OS2, MS2) = population[i+1]

        if random.random() < pc:
            (oOS1, oOS2) = crossoverOS(OS1, OS2, parameters)
            (oMS1, oMS2) = crossoverMS(MS1, MS2)
            newPop.append((oOS1, oMS1))
            newPop.append((oOS2, oMS2))
        else:
            newPop.append((OS1, MS1))
            newPop.append((OS2, MS2))

        i = i + 2

    return newPop


# 4.3.3 Mutation operators
##########################

def swappingMutation(p):
    pos1 = random.randint(0, len(p) - 1)
    pos2 = random.randint(0, len(p) - 1)

    if pos1 == pos2:
        return p

    if pos1 > pos2:
        pos1, pos2 = pos2, pos1

    offspring = p[:pos1] + [p[pos2]] + \
          p[pos1+1:pos2] + [p[pos1]] + \
          p[pos2+1:]

    return offspring


def neighborhoodMutation(p):
    pos3 = pos2 = pos1 = random.randint(0, len(p) - 1)

    while p[pos2] == p[pos1]:
        pos2 = random.randint(0, len(p) - 1)

    while p[pos3] == p[pos2] or p[pos3] == p[pos1]:
        pos3 = random.randint(0, len(p) - 1)

    sortedPositions = sorted([pos1, pos2, pos3])
    pos1 = sortedPositions[0]
    pos2 = sortedPositions[1]
    pos3 = sortedPositions[2]

    e1 = p[sortedPositions[0]]
    e2 = p[sortedPositions[1]]
    e3 = p[sortedPositions[2]]

    permutations = list(itertools.permutations([e1, e2, e3]))
    permutation  = random.choice(permutations)

    offspring = p[:pos1] + [permutation[0]] + \
          p[pos1+1:pos2] + [permutation[1]] + \
          p[pos2+1:pos3] + [permutation[2]] + \
          p[pos3+1:]

    return offspring


def halfMutation(p, parameters):
    o = p
    jobs = parameters['jobs']

    size = len(p)
    r = int(size/2)

    positions = random.sample(range(size), r)

    i = 0
    for job in jobs:
        for op in job:
            if i in positions:
                o[i] = random.randint(0, len(op)-1)
            i = i+1

    return o


def mutationOS(p):
    if random.choice([True, False]):
        return swappingMutation(p)
    else:
        return neighborhoodMutation(p)


def mutationMS(p, parameters):
    return halfMutation(p, parameters)


def mutation(population, parameters):
    newPop = []

    for (OS, MS) in population:
        if random.random() < pm:
            oOS = mutationOS(OS)
            oMS = mutationMS(MS, parameters)
            newPop.append((oOS, oMS))
        else:
            newPop.append((OS, MS))

    return newPop



# Gantt

def draw_chart(data):
    nb_row = len(data.keys())
    print(data)
    pos = np.arange(0.5, nb_row * 0.5 + 0.5, 0.5)

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)

    index = 0
    max_len = []
    #print(sorted(data.items()))

    for machine, operations in (data.items()):
        for op in operations:
            max_len.append(op[1])
            c = random.choice(colors)
            rect = ax.barh((index * 0.5) + 0.5, op[1] - op[0], left=op[0], height=0.3, align='center',
                           edgecolor=c, color=c, alpha=0.8)

            # adding label
            width = int(rect[0].get_width())
            Str = "OP_{}".format(op[2])
            xloc = op[0] + 0.50 * width
            clr = 'black'
            align = 'center'

            yloc = rect[0].get_y() + rect[0].get_height() / 2.0
            ax.text(xloc, yloc, Str, horizontalalignment=align,
                            verticalalignment='center', color=clr, weight='bold',
                            clip_on=True)
        index += 1

    ax.set_ylim(ymin=-0.1, ymax=nb_row * 0.5 + 0.5)
    ax.grid(color='gray', linestyle=':')
    ax.set_xlim(0, max(10, max(max_len)))

    labelsx = ax.get_xticklabels()
    plt.setp(labelsx, rotation=0, fontsize=10)

    locsy, labelsy = plt.yticks(pos, data.keys())
    plt.setp(labelsy, fontsize=14)

    font = font_manager.FontProperties(size='small')
    ax.legend(loc=1, prop=font)

    ax.invert_yaxis()

    plt.title("Flexible Job Shop Solution")
    plt.savefig('gantt.svg')
    plt.show()

# Beginning

parameters = parse(path)

t0 = time.time()

# Initialize the Population
population = initializebPopulation(parameters)
gen = 1

# Evaluate the population
while not shouldTerminate(population, gen):
    # Genetic Operators
    population = selection(population, parameters)
    population = crossover(population, parameters)
    population = mutation(population, parameters)

    gen = gen + 1

sortedPop = sorted(population, key=lambda cpl: timeTaken(cpl, parameters))

t1 = time.time()
total_time = t1 - t0

print("best result: " + str(sortedPop[0][0]))
print("Finished in {0:.2f}s".format(total_time))


# Termination Criteria Satisfied ?
gantt_data = translate_decoded_to_gantt(decode(parameters, sortedPop[0][0], sortedPop[0][1]))

draw_chart(gantt_data)