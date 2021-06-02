import sys
import time
import random
import itertools
import numpy as np

# Config
popSize = 40   #400
maxGen = 10    #200
pr = 0.005
pc = 0.8
pm = 0.1
latex_export = False


path = "/Users/sophi/Downloads/GA_Sophia/data/mk04.fjs"


#Parser

def parse(path):
    file = open(path, 'r')

    firstLine = file.readline()
    firstLineValues = list(map(int, firstLine.split()[0:2]))
    print(firstLineValues)

    jobsNb = firstLineValues[0]
    machinesNb = firstLineValues[1]

    #genesNb = machinesNb*jobsNb
    #num_mutation_jobs = round(genesNb*pr)

    jobs = []

    for i in range(jobsNb):
        currentLine = file.readline()
        currentLineValues = list(map(int, currentLine.split())) #faz um vetor com cada linha
        print(currentLineValues)

        operations = []

        j = 1
        while j < len(currentLineValues):
            k = currentLineValues[j] #lê cada valor da linha (máquina e tempo)
            j = j+1

            operation = []

            for ik in range(k):
                machine = currentLineValues[j]
                j = j+1
                processingTime = currentLineValues[j]
                j = j+1

                operation.append({'machine': machine, 'processingTime': processingTime})
                #print(operation)

            operations.append(operation)

        jobs.append(operations)


    file.close()

    return {'machinesNb': machinesNb, 'jobs': jobs}


# ENCODING

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

#Initialize Population

def initializePopulation(parameters):
    gen1 = []

    for i in range(popSize):
        OS = generateOS(parameters)
        MS = generateMS(parameters)
        gen1.append((OS, MS))

    print("GENE 1")
    print(gen1)
    return gen1

# Termination
def shouldTerminate(population, gen):
    return gen > maxGen

'''----- generate initial population -----
Tbest = 999999999999999
best_list, best_obj = [], []
population_list = []
makespan_record = []
for i in range(popSize):
    nxm_random_num = list(np.random.permutation(geneNb))  # generate a random permutation of 0 to jobsNb*machinesNb-1
    population_list.append(nxm_random_num)  # add to the population_list
    for j in range(geneNb):
        population_list[i][j] = population_list[i][j] % num_job  # convert to job number format, every job appears m times

for n in range(num_iteration):
    Tbest_now = 99999999999
'''

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

#Beginning

parameters = parse(path)
#print(parameters)

t0 = time.time()
print(t0)
# Initialize the Population
population = initializePopulation(parameters)
#print(population)
gen = 1

#Process
while not shouldTerminate(population,gen):
    # Genetic Operators
    #population = selection(population, parameters)
    #population = crossover(population, parameters)
    #population = mutation(population, parameters)

    gen = gen + 1

#sortedPop = sorted(population, key=lambda cpl: timeTaken(cpl, parameters))

t1 = time.time()
total_time = t1 - t0

#print("best result: " + str(sortedPop[0][0]))
print("Finished in {0:.2f}s".format(total_time))
print("Makespan:")

# Termination Criteria Satisfied ?
#gantt_data = translate_decoded_to_gantt(decode(parameters, sortedPop[0][0], sortedPop[0][1]))

#draw_chart(gantt_data)