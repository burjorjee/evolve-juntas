"""
An evolution based algorithm that, scalably ( O(polylog n) queries, and
O(n polylog n)time, where n is the total number of attributes ) learns
the relevant attributes of k-juntas for small values of k given a
membership query oracle
"""

import numpy as np
from numpy import arange, logical_xor, logical_and, log2, logical_not, \
    flatnonzero, array, zeros, ones, array_equal
from numpy.random import rand, shuffle, seed, randint
import locale
import time
import traceback
try:
    from matplotlib.pylab import plot, axis, figure, hold, title, subplot
    from matplotlib.pyplot import xlabel, ylabel
except:
    pass


class KJunta(object):

    def __init__(self, n, juntas, hiddenFn, k=None, rngSeed=None):
        """
        Parameters
        ----------
        n : int
            number of inputs to the k-junta

        juntas : an int or a one dimensional array of integers
            if int:
                the number of juntas (whose indices will be chosen at random
            else:
                the indices of the junta

        hiddenFn : str or a one dimensional numpy array of boolean values
            if str:
                must be one of "and", "or", "parity", or "random"
            else:
                The rightmost column of the hidden function's truth table
                The hidden function specified must be minimal. i.e. it must
                not itself have a function hidden within it.
        k : int
            Advertised upperbound on the number of juntas
        """
        if not rngSeed:
            rngSeed = int(time.time())
        seed(rngSeed)
        self.rngSeed = rngSeed

        self.j = juntas if isinstance(juntas, int) else len(juntas)
        if self.j > n:
            raise Exception("The number of juntas (%s) must be less than or equal to n (%s)" % (self.j, n))

        if isinstance(hiddenFn, str):
            while True:
                self.hiddenFn = createBooleanFn(hiddenFn, self.j)
                try:
                    assertBooleanFnIsMinimal(self.hiddenFn, self.j)
                    break
                except BooleanFunctionIsNotMinimalException, e:
                    pass
                except Exception, e:
                    traceback.print_exc()
                    raise e
        else:
            self.hiddenFn = hiddenFn
            if len(self.hiddenFn) != 2**self.j:
                raise Exception("hiddenFn must be a list of exactly %s boolean values" % 2**self.j)
            assertBooleanFnIsMinimal(self.hiddenFn, self.j)

        if isinstance(juntas, int):
            x = arange(n)
            shuffle(x)
            self.juntas = array(sorted(x[:self.j]))
        else:
            self.juntas = juntas

        if k is None:
            self.k = self.j
        else:
            assert k >= self.j
            self.k = k

        self.n = n

def assertBooleanFnIsMinimal(booleanFn, numInputs):
    """
    Asserts that no boolean function is "hidden" within the given boolean function
    """
    inputValues2TruthTableRowNum = create_inputValues2TruthTableRowNum(numInputs)
    for i in xrange(numInputs):
        x, y = [slice(2)] * numInputs , [slice(2)] * numInputs
        x[i], y[i] = 0, 1
        if array_equal(booleanFn[inputValues2TruthTableRowNum[tuple(x)]],
                       booleanFn[inputValues2TruthTableRowNum[tuple(y)]]):
            raise BooleanFunctionIsNotMinimalException("The hidden function must be  minimal")


def create_inputValues2TruthTableRowNum(numInputs):
    """
    Returns an array a of dimension numInputs such that for any tuple
    of boolean values v = (x_1, x_2, ..., x_numInputs), a[v] returns the
    vertical zero indexed position of v in the truth table of a boolean
    function with numInputs inputs
    """
    def create_inputValues2TruthTableRowNumHelper(inputValues, idx):
        if idx == len(inputValues):
            inputValues2TruthTableRowNum[tuple(inputValues)] = (inputValues * multiplicands).sum()
        else:
            create_inputValues2TruthTableRowNumHelper(inputValues, idx+1)
            inputValues[idx] = 1
            create_inputValues2TruthTableRowNumHelper(inputValues, idx+1)
            inputValues[idx] = 0

    inputValues2TruthTableRowNum = zeros((2,)*numInputs, dtype = int)
    multiplicands = 2**arange(numInputs)
    create_inputValues2TruthTableRowNumHelper([0]*numInputs, 0)
    return inputValues2TruthTableRowNum


class BooleanFunctionIsNotMinimalException(Exception):
    def __init__(self, message):
        Exception(self, message)


class MembershipQueryKJuntaOracle(object):
    """
    A membership query oracle that internally queries a k-junta
    """
    def __init__(self, kJunta):
        """
        Parameters
        ----------
        kJunta : KJunta
            The KJunta
        """
        self.__kJunta = kJunta
        self.numQueriesAnswered = 0

        # precompute multiplicands used in the calculation of the
        # value of hidden function given any input
        self.multiplicands = 2**arange(len(kJunta.juntas))

    def query(self, queries):
        """
        Parameters
        ----------
        queries: an ndarray of booleans
            Each row is a query
        """
        numQueries, inputsPerQuery = queries.shape
        assert self.__kJunta.n == inputsPerQuery
        booleanFunctionTableRowIndices = \
            (queries[:, self.__kJunta.juntas] * self.multiplicands).sum(axis=1, dtype=int)
        results = self.__kJunta.hiddenFn[booleanFunctionTableRowIndices]

        self.numQueriesAnswered += numQueries
        return results


def learnJuntas(n, k, membershipQueryFn, threeWayMajorityVotingDepth, learnedJuntas,
                restriction=None, visualizer=None,  recursionDepth=0, crippleFactor=0):
    """
    Learn the relevant attributes of the specified k-junta

    Parameters
    ----------
    n : int
        The total number of boolean attributes
    k : int
        Advertised upper bound on the number of juntas
    membershipQueryFn : Python function
        The membership query function
    threeWayMajorityVotingDepth : int
        Recursion depth of three way majority voting
    learnedJuntas : set
        The juntas that have been learned
    restriction : list of (index, value) tuples
        A list of attribute indices and the (fixed) values assigned to them
    visualizer : Python function
        A function that visually displays the state of the GA populations
    recursionDepth : int
        The depth at which this function is called recursively by itself

    crippleFactor: float between 0.0 and 1.0
        A non zero value marginally decreases the false negative rate while
        greatly increasing the false positive rate. Cranking up this value
        allows one to demonstrate the usefulness of "downstream" error
        correcting routines at low values of n

    Returns
    -------
      : set of ints
         The learned junta
    """

    probMutation = 0.005 * (1-crippleFactor)
    numGens = 300
    popSize = 500
    if not restriction:
        restriction = []

    indentation = "\t"*recursionDepth

    hypotheses = zeros((3**threeWayMajorityVotingDepth, n), dtype=bool)
    for i in range(3**threeWayMajorityVotingDepth):
        juntas = set()
        restrictedQueryFn = createRestrictedFn(restriction, membershipQueryFn)
        print indentation + "======== Evolving Voter %s ========" % (i+1)
        ugas = [ugaEvolve(popSize, n, probMutation, restrictedQueryFn),
                ugaEvolve(popSize, n, probMutation, lambda x: logical_not(restrictedQueryFn(x)))]

        with Timer(indentation):
            for gen in xrange(numGens+1):
                oneFreqsList = [uga.next()["oneFreqs"] for uga in ugas]
                if visualizer:
                    visualizer(gen, *oneFreqsList)

        for oneFreqs in oneFreqsList:
            juntas.update(
                {i for i in flatnonzero(logical_not(logical_and(0.05 < oneFreqs, oneFreqs < 0.95)))
                 if not restriction or i not in zip(*restriction)[0]})

        print indentation + "Hypothesized junta = %s" % array(sorted(list(juntas)))
        print indentation + "#queries = " + locale.format("%d", 2 * popSize * numGens, grouping=True)
        hypotheses[i,list(juntas)] = 1

    print indentation + "========================"
    learnedhypothesis = recursive3WayMajority(*[hypotheses[i,:] for i in xrange(3**threeWayMajorityVotingDepth)])

    newlyLearnedJuntas = learnedhypothesis.nonzero()[0]
    print indentation + "LEARNED juntas = %s" % newlyLearnedJuntas
    if len(newlyLearnedJuntas) == 0 :
        print  indentation + "The function is not constant under the restriction %s, "\
            "but no junta could be learned.\nChances are the learning algorithm will fail. " % restriction
        return
    learnedJuntas.update(set(newlyLearnedJuntas))
    for augmentedRestriction in exhaustivelyAugmentRestriction(restriction, learnedJuntas):
        if len(learnedJuntas) >= k:
            print indentation + "Total #juntas learned >= k. Exiting..."
            break
        print indentation + "Checking for constancy under the %s restriction %s" % \
            ("empty" if not augmentedRestriction else "following",
             "\n"+indentation+str(augmentedRestriction) if augmentedRestriction else "")
        restrictedQueryFn = createRestrictedFn(augmentedRestriction, membershipQueryFn)
        if iskJuntaConstant(n=n, k=k-len(augmentedRestriction), kJunta=restrictedQueryFn):
            print indentation + "Function seems constant under the restriction. Moving on..."
            continue
        print indentation + "Function is NOT constant. " \
                            "Learning its juntas under the restriction..."
        learnJuntas(n, k,
                    membershipQueryFn,
                    threeWayMajorityVotingDepth,
                    learnedJuntas,
                    augmentedRestriction,
                    visualizer,
                    recursionDepth+1,
                    crippleFactor)


def ugaEvolve(popSize, bitstringLength, probMutation, fitnessFn):
    """
    A Python generator that evolves a population of bitstrings on the
    specified fitness function using uniform crossover,
    and a per bit probability of mutation given by probMutation.
    Selection is as follows:
    if at least one bitstring evaluates to True under the fitnessFn:
        Pick 2*popSize parents as close as possible to evenly
        from amongst the bistrings that evaluate to True under
        the fitnessFn
    else:
        Pick 2*popSize parents evenly from amongst the bitstrings
        in the existing population (each bitstring gets picked twice)


    Parameters
    ----------
    popSize : int
        The size of the population
    bitstringLength : int
        The length of each bitstring in the population
    probMutation : float
        The probability that a given bit will be mutated
        in a single generation
    fitnessFn : Python function
        The fitness function to be used
    """

    # initialize a population of bitstrings drawn uniformly at random
    pop = rand(popSize,bitstringLength) < 0.5
    recombinationMasksRepo = rand(popSize * 10, bitstringLength) < 0.5
    mutationMasksRepo = rand(popSize * 10, bitstringLength) < probMutation
    while True:

        # evaluate fitness of each bitstring in the population
        fitnessVals = fitnessFn(pop)

        # calculate the oneFrequency of all bitstringLength attributes
        oneFreqs = pop.sum(axis=0, dtype=float) / popSize
        yield dict(oneFreqs=oneFreqs, fitnessVals=fitnessVals)

        # select parents
        nonZeroFitnessValIndices = fitnessVals.nonzero()[0]
        if len(nonZeroFitnessValIndices):
            parentIndices = nonZeroFitnessValIndices[arange(2 * popSize, dtype=int) % len(nonZeroFitnessValIndices)]
            shuffle(parentIndices)
        else:
            parentIndices = arange(2 * popSize) % popSize
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
    returns a function that for any input sets the attributes
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

def iskJuntaConstant(n, k, kJunta):
    """
    is the specified kJunta over n attributes constant?
    """
    for x in xrange(2*2**k):
        queries = rand(1000, n) < 0.5
        results = kJunta(queries)
        if np.any(results) != np.all(results):
            return False
    return True

def exhaustivelyAugmentRestriction(currentRestriction, attributesToRestrict):
    """
    Generator function that exhaustively augments the currentRestriction with
    assignments of boolean values to as yet unrestricted attributes in
    attributesToRestrict
    Note: attributesToRestrict is not expected to be static between yields

    Parameters
    ----------
    currentRestriction: a list of (int, boolean) pairs
        Each pair contains an attribute index and the value to which it is restricted
    attributesToRestrict: set of ints
        Attributes in need of restriction. May include attributes that
        appear in currentRestriction (they will be ignored)

    """
    unrestrictedAttributesToRestrict = attributesToRestrict - \
                                   set(zip(*currentRestriction)[0] if currentRestriction else set([]))
    if not unrestrictedAttributesToRestrict:
            yield currentRestriction
    else:
        indexToAdd = min(unrestrictedAttributesToRestrict)
        currentRestriction.append((indexToAdd, 0))
        for augmentedRestriction in exhaustivelyAugmentRestriction(currentRestriction, attributesToRestrict):
            yield augmentedRestriction
        currentRestriction.pop()
        currentRestriction.append((indexToAdd, 1))
        for augmentedRestriction in exhaustivelyAugmentRestriction(currentRestriction, attributesToRestrict):
            yield augmentedRestriction
        currentRestriction.pop()

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

def createBooleanFn(fnName, numInputs):
    """
    Returns an array of 2^numInputs boolean that give the
    rightmost (output) column of the truth table of the
    specified function over the specified number of inputs

    Parameters
    ----------
    name : str
        Should be one of "and", "or", "parity", or "random"
    numInputs : int
        The number of inputs to the boolean function
    """
    fnName = fnName.lower()
    if fnName == "and":
        function = zeros(2**numInputs, dtype=bool)
        function[-1] = True
        return function
    elif fnName == "or":
        function = ones(2**numInputs, dtype=bool)
        function[0] = False
        return function
    elif fnName == "parity":
        function = zeros(2**numInputs, dtype=int)
        temp = arange(2**numInputs)
        for i in xrange(numInputs-1, -1, -1):
            function += temp/(2**i)
            temp %= 2**i
        function %= 2
        function = function.astype(bool)
    elif fnName == "random":
        return rand(2**numInputs)<0.5
    else:
        raise Exception("Unknown boolean function name %s" % fnName)
    return function

def recoverJuntas(kJunta, threeWayMajorityVotingDepth=0, rngSeed=None, visualize=False, crippleFactor=0):
    """
    Set up a leaning juntas problem. Then learn the juntas.
    Finally, check if junta recovery was successful

    Parameters
    ----------
    kJunta : KJunta
        The kJunta whose relevant attributes must be recovered
    threeWayMajorityVotingDepth :
        The number of levels of three-way majority voting to use
        to elminiate errors a hypothesized list of juntas
    rngSeed : int
        The random number generator seed for the learning algorithm
    visualize : boolean
        If true, visualize evolutionary. Slows down the learning
        algorithm; especially for large values of kJunta.n

    crippleFactor: float between 0.0 and 1.0
        A non zero value marginally decreases the false negative rate while
        greatly increasing the false positive rate. Cranking up this value
        allows one to demonstrate the usefulness of "downstream" error
        correcting routines at low values of kJunta.n
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
            plot(kJunta.juntas, oneFreqs[i][kJunta.juntas], 'r.')
            axis([1, kJunta.n, 0, 1])
            ylabel("1-Frequency")
            if i==0:
                title("Generation = %s " % genNum)
        ax.set_xlabel("Attribute")
        f.canvas.draw()
        f.show()

    if rngSeed is None:
        rngSeed = int(time.time())
    seed(rngSeed)

    k = kJunta.k
    print "k-junta creation RNG seed  : %s" % kJunta.rngSeed
    print "Learning algorithm RNG seed: %s" % rngSeed
    print "n = %s" % kJunta.n
    print "k = %s" % k
    print "Hidden boolean function = %s" % ("".join(["1" if b else "0" for b in kJunta.hiddenFn[:128].tolist()]) +
        ("...followed by %s bits" % locale.format("%d", 2**kJunta.j-128, grouping=True) if kJunta.j > 7 else ""))
    print "True juntas =   %s" % kJunta.juntas


    #initialize the oracle
    oracle = MembershipQueryKJuntaOracle(kJunta)

    with Timer() as t:
        print "Checking for constancy..."
        if iskJuntaConstant(kJunta.n, k, oracle.query):
            print "Function seems to be constant."
            recoveredJuntas = []
        else:
            print "Function is NOT constant. Learning its juntas..."
            # learn the juntas
            recoveredJuntas = set()
            learnJuntas(kJunta.n, k,
                        membershipQueryFn=oracle.query,
                        learnedJuntas=recoveredJuntas,
                        threeWayMajorityVotingDepth=threeWayMajorityVotingDepth,
                        visualizer=visualizeGen if visualize else None,
                        crippleFactor=crippleFactor)
            recoveredJuntas = array(sorted(recoveredJuntas))

        match = (len(kJunta.juntas) == len(recoveredJuntas) and all(kJunta.juntas == recoveredJuntas))
        print "_______________________________________________"
        print "True juntas      = %s" % kJunta.juntas
        print "Recovered juntas = %s" % recoveredJuntas
        print "Match? : %s" %  match
        print
        print "Total number of queries = " + locale.format("%d", oracle.numQueriesAnswered, grouping=True)
    return match


def checkAllMinimalHiddenFnsWithMInputs(m):
    def generateAllFns(fn, idx):
        if idx == len(fn):
            yield fn
        else:
            for x in generateAllFns(fn, idx+1):
                yield x
            fn[idx] = 1
            for x in generateAllFns(fn, idx+1):
                yield x
            fn[idx] = 0

    unrecoveredFns = []
    for fn in generateAllFns(zeros(2**m, dtype=int), 0):
        try:
            if not recoverJuntas(KJunta(10, juntas=m, hiddenFn=fn)):
                unrecoveredFns.append(fn)
        except BooleanFunctionIsNotMinimalException:
            pass
    print "Unrecovered Functions"
    print "---------------------"
    print unrecoveredFns


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
    recoverJuntas(KJunta(n=100, juntas=7, hiddenFn="random"))
    raw_input("Press Enter to end")