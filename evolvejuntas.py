"""
An evolution based algorithm that scalably (w.r.t. the total number of variables) learns
the relevant variables of $k$-juntas given a membership query oracle when $k$ is small
"""
from string import replace

import numpy as np
from numpy import arange, empty, histogram, logical_xor, logical_and, \
    squeeze, r_, logical_not, flatnonzero, array, zeros, ones, cumsum
from numpy.random import rand, shuffle, seed, randint
import locale
import time
try:
    from matplotlib.pylab import plot, axis, figure, hold, title, subplot
    from matplotlib.pyplot import xlabel, ylabel
except:
    pass

def learnJuntas(k, n, kJunta, votingTreeDepth, learnedJuntas, assignment=None, visualizer=None,  recursionDepth=0):
    """
    learns the relevant variables of the specified k-junta
    @param k: upper bound on the number of juntas
    @param n: the total number of boolean variables
    @param kJunta: the k-junta
    @param votingTreeDepth: recursion depth of three way majority voting
    @param learnedJuntas: the juntas that have been learned
    @param assignment:list of (index, value) tuples specifying  variable indices and their fixed values
    @param visualizer: function used to visualize GA populations
    @param recursionDepth: depth at which this function is called recursively
    @returns a set of learned junta indices
    """

    probMutation = 0.004
    numGens = 300
    popSize = 500
    if not assignment:
        assignment = []

    indentation = "\t"*recursionDepth
    
    hypotheses = zeros((3**votingTreeDepth, n), dtype=bool)
    for i in range(3**votingTreeDepth):
        juntaIndices = set()
        assignmentEnforcingQueryFn = createAssignmentEnforcingBooleanFn(assignment, kJunta)
        print indentation + "======== run %s ========" % (i+1)
        ugas = [ugaEvolve(popSize, n, probMutation, assignmentEnforcingQueryFn),
                ugaEvolve(popSize, n, probMutation, lambda x: logical_not(assignmentEnforcingQueryFn(x)))]

        with Timer(indentation):
            for gen in xrange(numGens):
                oneFreqsList = [uga.next()["oneFreqs"] for uga in ugas]
                if visualizer:
                    visualizer(gen, *oneFreqsList)

        for oneFreqs in oneFreqsList:
            juntaIndices.update(
                {i for i in flatnonzero(logical_not(logical_and(0.05 < oneFreqs, oneFreqs < 0.95)))
                 if not assignment or i not in zip(*assignment)[0]})

        print indentation + "Hypothesized indices = %s" % array(sorted(list(juntaIndices)))
        print indentation + "#queries = " + locale.format("%d", 2 * popSize * numGens, grouping=True)
        hypotheses[i,list(juntaIndices)] = 1

    print indentation + "========================"
    learnedhypothesis = recursive3WayMajority(*[hypotheses[i,:] for i in xrange(3**votingTreeDepth)])

    newlyLearnedJuntas = learnedhypothesis.nonzero()[0]
    print indentation + "LEARNED juntas = %s" % newlyLearnedJuntas
    if len(newlyLearnedJuntas) == 0 :
        s = "The function is not constant under the following assignment, "\
            "but no junta could be learned. " \
            "In other words, the algorithm has failed. " \
            "Assignment = %s" % assignment
        raise Exception(s)
    learnedJuntas.update(set(newlyLearnedJuntas))
    for augmentedAssignment in exhaustivelyAugmentAssignment(assignment, learnedJuntas):
        if len(learnedJuntas) >= k:
            print indentation + "#juntas learned >= k. Exiting..."
            break
        print indentation + "Checking for constancy under the %s assignment %s" % \
            ("empty" if not augmentedAssignment else "", augmentedAssignment)
        assignmentEnforcingQueryFn = createAssignmentEnforcingBooleanFn(augmentedAssignment, kJunta)
        if iskJuntaConstant(k=k-len(augmentedAssignment), n=n, kJunta=assignmentEnforcingQueryFn):
            print indentation + "Function seems constant under the assignment. Nothing to learn here; moving on..."
            continue
        print indentation + "Function is NOT constant under the assignment. " \
                            "Learning its juntas under the assignment..."
        learnJuntas(k, n,
                    kJunta,
                    votingTreeDepth,
                    learnedJuntas,
                    augmentedAssignment,
                    visualizer,
                    recursionDepth+1)


def ugaEvolve(popSize, bitstringLength, probMutation, fitnessFn):
    """
    Evolve a population of bitstrings on the specified fitness function
    using fitness proportionate stochastic universal sampling selection,
    uniform crossover, and the per bit mutation with the specified
    mutation rate

    @param popSize: the size of the population
    @param bitstringLength: the length of each bitstring in the population
    @param probMutation: the probability that a given bit will be mutated
                         in a single generation
    """

    # initialize a population of bitstrings drawn uniformly at random
    pop = rand(popSize,bitstringLength) < 0.5
    recombinationMasksRepo = rand(popSize * 10, bitstringLength) < 0.5
    mutationMasksRepo = rand(popSize * 10, bitstringLength) < probMutation
    ctr = 0
    while True:
        ctr += 1

        # evaluate fitness of each bitstring in the population
        fitnessVals = fitnessFn(pop)

        #calculate the oneFrequency of all bitstringLength attributes
        oneFreqs = pop.sum(axis=0, dtype=float) / popSize
        yield dict(oneFreqs=oneFreqs, fitnessVals=fitnessVals)

        # use fitness proportional selection to select 2*popSize parents
        totalFitness = sum(fitnessVals)
        cumNormFitnessVals = squeeze(cumsum(fitnessVals).astype(float) / totalFitness)
        parentIndices = empty(2 * popSize, dtype=int)
        markers = arange(2 * popSize, dtype=float) / (2 * popSize) + rand()
        markers[markers>1]-=1
        numParentsWithIndex, _ = histogram(markers, bins=r_[0, cumNormFitnessVals])
        j = 0
        for i in range(popSize):
            parentIndices[j:j+numParentsWithIndex[i]] = i
            j += numParentsWithIndex[i]
        shuffle(parentIndices)

        # recombine the parents using uniform crossover to generate
        # one offspring per parent pair
        recombinationMasks = recombinationMasksRepo[randint(popSize*10, size=popSize), :]
        newPop = pop[parentIndices[:popSize], :]
        newPop[recombinationMasks] = pop[parentIndices[popSize:], :][recombinationMasks]

        # mutate the offspring
        mutationMasks = mutationMasksRepo[randint(popSize*10, size=popSize), :]
        pop = logical_xor(newPop, mutationMasks)


def createAssignmentEnforcingBooleanFn(assignment, booleanFn):
    """
    Takes a boolean function booleanFn and an assignment as input and
    returns a function that for any input sets the variables
    specified in the assignment to the assigned value before querying
    booleanFn with the input and returning the output
    """
    def assignmentEnforcingBooleanFn(bitstrings):
        bitstrings[:, zip(*assignment)[0]] = zip(*assignment)[1]
        return booleanFn(bitstrings)
    if assignment:
        return assignmentEnforcingBooleanFn
    else:
        return booleanFn

def iskJuntaConstant(k, n, kJunta):
    """
    is the specified kJunta over n variables constant?
    """
    queries = rand(2 * 2**k, n) < 0.5
    results = kJunta(queries)
    if np.any(results) != np.all(results):
        return False
    return True

def exhaustivelyAugmentAssignment(assignment, learnedJuntas):
    """
    Generator function that returns the given assignment with all possible assignments of
    boolean values to the the indices in indicesToAddToAssignment.

    """
    #import pdb; pdb.set_trace()
    unassignedIndices = learnedJuntas - set(zip(*assignment)[0] if assignment else [])
    if not unassignedIndices:
            yield assignment
    else:
        indexToAdd = min(unassignedIndices)
        assignment.append((indexToAdd, 0))
        for augmentedAssignment in exhaustivelyAugmentAssignment(assignment, learnedJuntas):
            yield augmentedAssignment
        assignment.pop()
        assignment.append((indexToAdd, 1))
        for augmentedAssignment in exhaustivelyAugmentAssignment(assignment, learnedJuntas):
            yield augmentedAssignment
        assignment.pop()

def threeWayMajority(hypothesis1, hypothesis2, hypothesis3):
    return hypothesis1.astype("int16") + \
           hypothesis2.astype("int16") + \
           hypothesis3.astype("int16") >= 2

def recursive3WayMajority(*hypotheses):
    num = len(hypotheses)
    if num == 1:
        return hypotheses[0]
    assert num % 3 == 0
    if num == 3:
        return threeWayMajority(*hypotheses)
    else:
        return threeWayMajority(recursive3WayMajority(*hypotheses[:num/3]),
                                recursive3WayMajority(*hypotheses[num/3:2*num/3]),
                                recursive3WayMajority(*hypotheses[2*num/3:]))

def createBooleanFunction(name, numInputs):
    """
    Returns an array of 2^numInputs boolean values representing
    a boolean function over numInputs inputs
    @param name: Should be one of "and", "or", "parity", or "random"
    @param numInputs: The number of inputs to the boolean function
    """
    name = name.lower()
    if name == "and":
        function = zeros(2**numInputs, dtype=bool)
        function[-1] = True
        return function
    elif name == "or":
        function = ones(2**numInputs, dtype=bool)
        function[0] = False
        return function
    elif name == "parity":
        function = zeros(2**numInputs, dtype=int)
        temp = arange(2**numInputs)
        for i in xrange(numInputs-1, -1, -1):
            function += temp/(2**i)
            temp %= 2**i
        function %= 2
        function = function.astype(bool)
    elif name == "random":
        return rand(2**numInputs)<0.5
    else:
        raise Exception("Unknown boolean function name %s" % name)
    return function

class MembershipQueryKJuntaOracle(object):
    """
    A membership query oracle whose that internally queries a k-junta
    """
    def __init__(self, k, n, juntaIndices, hiddenFn):
        """
        @param k: #juntas
        @param n: #inputs to the k-junta
        @param hiddenFn : an array of 2^k boolean values
        """
        assert k <= n
        assert len(juntaIndices) == k
        assert len(hiddenFn) == 2**k
        self.n = n
        self.k = k
        self.juntaIndices = juntaIndices
        self.hiddenFn = hiddenFn
        self.numQueriesAnswered = 0

        # precompute multiplicands used in the calculation of the
        # value of hidden function given any input
        self.multiplicands = empty((1,k), dtype="u4")
        for i in xrange(k):
            self.multiplicands[0, i] = 2**(k-i-1)

    def query(self, queries):
        """
        @param queries: an array of queries (each row is a single query)
        """
        numQueries, _ = queries.shape

        booleanFunctionTableRowIndices = \
            (queries[:, self.juntaIndices] * self.multiplicands).sum(axis=1, dtype=int)
        results = self.hiddenFn[booleanFunctionTableRowIndices]

        self.numQueriesAnswered += numQueries
        return results


def recoverJuntas(k, n,
                  hiddenFnName="random",
                  juntaIndices=None,
                  juntaCreationRngSeed=None,
                  algoMajorityVotingDepth=1,
                  algoRngSeed=None,
                  algoVisualize=False,
                  ):
    """
    Set up a leaning juntas problem. Then learn the juntas.

    Parameters with an algo prefix are algorithm specific.
    Other parameters specify k-junta creation.
    """
    def visualizeGen(genNum, *oneFreqs):
        numPops = len(oneFreqs)
        f = figure(1)
        ax = None
        for i in xrange(numPops):
            hold(False)
            ax = subplot(2,1,i+1)
            plot(oneFreqs[i], 'b.')
            hold(True)
            plot(juntaIndices, oneFreqs[i][juntaIndices], 'r.')
            axis([1, n, 0, 1])
            ylabel("1-Frequency")
            if i==0:
                title("Generation = %s " % genNum)
        ax.set_xlabel("Attribute")
        f.canvas.draw()
        f.show()

    if juntaCreationRngSeed is None:
        juntaCreationRngSeed = int(time.time())
    seed(juntaCreationRngSeed)

    hiddenFn = createBooleanFunction(hiddenFnName, k)

    if juntaIndices:
        juntaIndices = array(sorted(juntaIndices))
    else:
        x = arange(n)
        shuffle(x)
        juntaIndices = array(sorted(x[:k]))

    if algoRngSeed is None:
        algoRngSeed = int(time.time())
    seed(algoRngSeed)

    print "k-junta creation RNG seed  : %s" % juntaCreationRngSeed
    print "Learning algorithm RNG seed: %s" %algoRngSeed
    print "n = %s" % n
    print "k = %s" % k
    print "Hidden boolean function = %s" % ("".join(["1" if b else "0" for b in hiddenFn[:128].tolist()]) +
                                               ("...followed by %s bits" % locale.format("%d", 2**k-128, grouping=True)
                                                    if len(hiddenFn)>128 else ""))
    print "True junta indices =   %s" % juntaIndices


    #initialize the oracle
    oracle = MembershipQueryKJuntaOracle(k, n,
                                         juntaIndices,
                                         hiddenFn)

    print "Checking for constancy..."
    if iskJuntaConstant(k, n, oracle.query):
        print "Function seems to be constant."
        recoveredJuntaIndices = []
    else:
        print "Function is NOT constant. Learning its juntas..."
        # learn the juntas
        recoveredJuntaIndices = set()
        learnJuntas(k, n,
                    kJunta=oracle.query,
                    learnedJuntas=recoveredJuntaIndices,
                    votingTreeDepth=algoMajorityVotingDepth,
                    visualizer=visualizeGen if algoVisualize else None)
        recoveredJuntaIndices = array(sorted(recoveredJuntaIndices))

    print "_______________________________________________"
    print "True junta indices      = %s" % juntaIndices
    print "Recovered junta indices = %s" % recoveredJuntaIndices
    print "Match? : %s" % (len(juntaIndices) == len(recoveredJuntaIndices) and
            all(juntaIndices == recoveredJuntaIndices))
    print
    print "Total number of queries = " + locale.format("%d", oracle.numQueriesAnswered, grouping=True)
    print

class Timer:

    def __init__(self,indentation=""):
        self.indentation = indentation

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        print self.indentation+"Elapsed wall-clock time = %.03f seconds" % self.interval

np.set_printoptions(linewidth=100000)

if __name__ == "__main__":
    recoverJuntas(k=7, n=100)
    raw_input("Press Enter to end")