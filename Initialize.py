import random
import Constants as const


def generateOS(parameters):
    jobs = parameters['jobs']

    OS = []
    i = 0
    for job in jobs:
        for op in job:
            OS.append(i)
        # cada job tem seu index
        # poderia ser algo tipo job.map((_, i) => i).forEach...
        i = i + 1

    random.shuffle(OS)

    return OS


def generateMS(parameters):
    jobs = parameters['jobs']

    MS = []
    for job in jobs:
        for op in job:
            randomMachine = random.randint(0, len(op) - 1)
            MS.append(randomMachine)

    return MS


def initializePopulation(parameters):
    gen1 = []

    for i in range(const.POP_SIZE):
        OS = generateOS(parameters)
        MS = generateMS(parameters)
        gen1.append((OS, MS))

    return gen1
