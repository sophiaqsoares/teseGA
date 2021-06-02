import operator

import Constants as const
import random


def selection(population, parameters):
    # o index corresponde à posição da população original, e o valor é resultado da função de fitness
    genesWithFitness = []

    for index, gene in enumerate(population):
        fitness = timeTaken(gene)
        genesWithFitness.insert(index, (gene, fitness))

    keptPopSize = int(const.REMAIN_PERCENTAGE * len(population))
    genesWithFitness.sort(key=operator.itemgetter(1))
    newPop = genesWithFitness[:keptPopSize]
    newPop = newPop + newPop
    rest = genesWithFitness[keptPopSize:]

    numeroAindaSemNome = len(population) - len(newPop)
    quantosPegar = int(numeroAindaSemNome * 0.5) #TODO deve ser uma porcentagem bem menor
    newPop = newPop + rest[keptPopSize:keptPopSize+quantosPegar]

    outroResto = len(population) - len(newPop)

    while outroResto > 0:
        outroResto = outroResto - 1
        newPop.append(random.choice(genesWithFitness))

    return [x[0] for x in newPop]


def timeTaken(lista):
    return sum(lista)

# def timeTaken(os_ms, pb_instance):
#     (os, ms) = os_ms
#     decoded = decode(pb_instance, os, ms)
#
#     # Getting the max for each machine
#     max_per_machine = []
#     for machine in decoded:
#         max_d = 0
#         for job in machine:
#             end = job[3] + job[1]
#             if end > max_d:
#                 max_d = end
#         max_per_machine.append(max_d)
#     return max(max_per_machine)
#
#
# def decode(pb_instance, os, ms):
#     o = pb_instance['jobs']
#     machine_operations = [[] for i in range(pb_instance['machinesNb'])]
#
#     ms_s = split_ms(pb_instance, ms)  # machine for each operations
#
#     indexes = [0] * len(ms_s)
#     start_task_cstr = [0] * len(ms_s)
#
#     # Iterating over OS to get task execution order and then checking in
#     # MS to get the machine
#     for job in os:
#         index_machine = ms_s[job][indexes[job]]
#         machine = o[job][indexes[job]][index_machine]['machine']
#         prcTime = o[job][indexes[job]][index_machine]['processingTime']
#         start_cstr = start_task_cstr[job]
#
#         # Getting the first available place for the operation
#         start = find_first_available_place(start_cstr, prcTime, machine_operations[machine - 1])
#         name_task = "{}-{}".format(job, indexes[job] + 1)
#
#         machine_operations[machine - 1].append((name_task, prcTime, start_cstr, start))
#
#         # Updating indexes (one for the current task for each job, one for the start constraint
#         # for each job)
#         indexes[job] += 1
#         start_task_cstr[job] = (start + prcTime)
#
#     return machine_operations
#
#
# def split_ms(pb_instance, ms):
#     jobs = []
#     current = 0
#     for index, job in enumerate(pb_instance['jobs']):
#         jobs.append(ms[current:current + len(job)])
#         current += len(job)
#     return jobs
#
#
# def find_first_available_place(start_ctr, duration, machine_jobs):
#     max_duration_list = []
#     max_duration = start_ctr + duration
#
#     # max_duration is either the start_ctr + duration or the max(possible starts) + duration
#     if machine_jobs:
#         for job in machine_jobs:
#             max_duration_list.append(job[3] + job[1])  # start + process time
#
#         max_duration = max(max(max_duration_list), start_ctr) + duration
#
#     machine_used = [True] * max_duration
#
#     # Updating array with used places
#     for job in machine_jobs:
#         start = job[3]
#         long = job[1]
#         for k in range(start, start + long):
#             machine_used[k] = False
#
#     # Find the first available place that meets constraint
#     for k in range(start_ctr, len(machine_used)):
#         if is_free(machine_used, k, duration):
#             return k
#
#
# def is_free(tab, start, duration):
#     for k in range(start, start + duration):
#         if not tab[k]:
#             return False
#     return True
