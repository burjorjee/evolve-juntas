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

def learnJuntas(k, n, membershipQueryFn, votingTreeDepth, learnedJuntas, restriction=None, visualizer=None,  recursionDepth=0):
    """
    Learn the relevant variables of the specified k-junta

    Parameters
    ----------
    k : int
        Upper bound on the number of juntas
    n : int
        The total number of boolean variables
    membershipQueryFn : Python function
        The membership query function
    votingTreeDepth : int
        Recursion depth of three way majority voting
    learnedJuntas : set
        The juntas that have been learned
    restriction : list of (index, value) tuples
        A list of variable indices and the (fixed) values assigned to them
    visualizer : Python function
        A function that visually displays the state of the GA populations
    recursionDepth : int
        The depth at which this function is called recursively by itself

    Returns
    -------
      : set
         The learned junta indices
    """

    probMutation = 0.004
    numGens = 300
    popSize = 500
    if not restriction:
        restriction = []

    indentation = "\t"*recursionDepth
    
    hypotheses = zeros((3**votingTreeDepth, n), dtype=bool)
    for i in range(3**votingTreeDepth):
        juntaIndices = set()
        restrictedQueryFn = createRestrictedFn(restriction, membershipQueryFn)
        print indentation + "======== Round %s ========" % (i+1)
        ugas = [ugaEvolve(popSize, n, probMutation, restrictedQueryFn),
                ugaEvolve(popSize, n, probMutation, lambda x: logical_not(restrictedQueryFn(x)))]

        with Timer(indentation):
            for gen in xrange(numGens):
                oneFreqsList = [uga.next()["oneFreqs"] for uga in ugas]
                if visualizer:
                    visualizer(gen, *oneFreqsList)

        for oneFreqs in oneFreqsList:
            juntaIndices.update(
                {i for i in flatnonzero(logical_not(logical_and(0.05 < oneFreqs, oneFreqs < 0.95)))
                 if not restriction or i not in zip(*restriction)[0]})

        print indentation + "Hypothesized indices = %s" % array(sorted(list(juntaIndices)))
        print indentation + "#queries = " + locale.format("%d", 2 * popSize * numGens, grouping=True)
        hypotheses[i,list(juntaIndices)] = 1

    print indentation + "========================"
    learnedhypothesis = recursive3WayMajority(*[hypotheses[i,:] for i in xrange(3**votingTreeDepth)])

    newlyLearnedJuntas = learnedhypothesis.nonzero()[0]
    print indentation + "LEARNED juntas = %s" % newlyLearnedJuntas
    if len(newlyLearnedJuntas) == 0 :
        print  indentation + "The function is not constant under the restriction %s, "\
            "but no junta could be learned.\nIn other words, the algorithm has failed. " % restriction
        return
    learnedJuntas.update(set(newlyLearnedJuntas))
    for augmentedRestriction in exhaustivelyAugmentRestriction(restriction, learnedJuntas):
        if len(learnedJuntas) >= k:
            print indentation + "#juntas learned >= k. Exiting..."
            break
        print indentation + "Checking for constancy under the %s restriction %s" % \
            ("empty" if not augmentedRestriction else "following",
             "\n"+indentation+str(augmentedRestriction) if augmentedRestriction else "")
        restrictedQueryFn = createRestrictedFn(augmentedRestriction, membershipQueryFn)
        if iskJuntaConstant(k=k-len(augmentedRestriction), n=n, kJunta=restrictedQueryFn):
            print indentation + "Function seems constant under the restriction. Moving on..."
            continue
        print indentation + "Function is NOT constant. " \
                            "Learning its juntas under the restriction..."
        learnJuntas(k, n,
                    membershipQueryFn,
                    votingTreeDepth,
                    learnedJuntas,
                    augmentedRestriction,
                    visualizer,
                    recursionDepth+1)


def ugaEvolve(popSize, bitstringLength, probMutation, fitnessFn):
    """
    Evolve a population of bitstrings on the specified fitness function
    using fitness proportionate stochastic universal sampling selection,
    uniform crossover, and the per bit mutation with the specified
    mutation rate

    Parameters
    ----------
    popSize : int
        The size of the population
    bitstringLength : int
        The length of each bitstring in the population
    probMutation : float
        The probability that a given bit will be mutated
        in a single generation

    Returns
    -------
        A generator object
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


def createRestrictedFn(restriction, fn):
    """
    Takes a function fn and a restriction as input and
    returns a function that for any input sets the variables
    specified in the restriction to the assigned values,
    queries the function and returns the output
    """
    def restrictedFn(inputs):
        inputs[:, zip(*restriction)[0]] = zip(*restriction)[1]
        return fn(inputs)

    if restriction:
        return restrictedFn
    else:
        return fn

def iskJuntaConstant(k, n, kJunta):
    """
    is the specified kJunta over n variables constant?
    """
    for x in xrange(2*2**k):
        queries = rand(1000, n) < 0.5
        results = kJunta(queries)
        if np.any(results) != np.all(results):
            return False
    return True

def exhaustivelyAugmentRestriction(restriction, learnedJuntas):
    """
    Generator function that yields the given restriction exhaustively augmented with
    assignments of boolean values to learned but as yet unrestricted juntas

    """
    indicesToRestrict = learnedJuntas - set(zip(*restriction)[0] if restriction else [])
    if not indicesToRestrict:
            yield restriction
    else:
        indexToAdd = min(indicesToRestrict)
        restriction.append((indexToAdd, 0))
        for augmentedRestriction in exhaustivelyAugmentRestriction(restriction, learnedJuntas):
            yield augmentedRestriction
        restriction.pop()
        restriction.append((indexToAdd, 1))
        for augmentedRestriction in exhaustivelyAugmentRestriction(restriction, learnedJuntas):
            yield augmentedRestriction
        restriction.pop()

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

    Parameters
    ----------
    name : str
        Should be one of "and", "or", "parity", or "random"
    numInputs : int
        The number of inputs to the boolean function
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
        Parameters
        ----------
            queries: an array of queries (each row is a single query)
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
                  majorityVotingDepth=0,
                  algoRngSeed=None,
                  visualize=False,
                  ):
    """
    Set up a leaning juntas problem. Then learn the juntas.

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

    hiddenFnName = createBooleanFunction(hiddenFnName, k)

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
    print "Hidden boolean function = %s" % ("".join(["1" if b else "0" for b in hiddenFnName[:128].tolist()]) +
                                               ("...followed by %s bits" % locale.format("%d", 2**k-128, grouping=True)
                                                    if len(hiddenFnName)>128 else ""))
    print "True junta indices =   %s" % juntaIndices


    #initialize the oracle
    oracle = MembershipQueryKJuntaOracle(k, n,
                                         juntaIndices,
                                         hiddenFnName)

    with Timer() as t:
        print "Checking for constancy..."
        if iskJuntaConstant(k, n, oracle.query):
            print "Function seems to be constant."
            recoveredJuntaIndices = []
        else:
            print "Function is NOT constant. Learning its juntas..."
            # learn the juntas
            recoveredJuntaIndices = set()
            learnJuntas(k, n,
                        membershipQueryFn=oracle.query,
                        learnedJuntas=recoveredJuntaIndices,
                        votingTreeDepth=majorityVotingDepth,
                        visualizer=visualizeGen if visualize else None)
            recoveredJuntaIndices = array(sorted(recoveredJuntaIndices))

        match = (len(juntaIndices) == len(recoveredJuntaIndices) and all(juntaIndices == recoveredJuntaIndices))
        print "_______________________________________________"
        print "True junta indices      = %s" % juntaIndices
        print "Recovered junta indices = %s" % recoveredJuntaIndices
        print "Match? : %s" %  match
        print
        print "Total number of queries = " + locale.format("%d", oracle.numQueriesAnswered, grouping=True)
    return match

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