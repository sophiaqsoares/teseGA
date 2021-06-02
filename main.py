from IO import readData
from Initialize import initializePopulation
from Selection import selection
import Constants as const
import random

def main():
    data = readData()
    # population = initializePopulation(data)
    population = [
        [4,1000],[1,1,1],[20,20,20],[300,200,2],[3,33],[3,34]
    ]
    gen = 1

    # while (gen < const.MAX_GENERATIONS):
    population = selection(population, data)
    gen = gen + 1



main()
