from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()



def joinFactors(factors: List[Factor]):
    """
    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))


    "*** YOUR CODE HERE ***"
    # Initialize sets for tracking variables.
    unconditioned_variables = set()
    conditioned_variables = set()

    # Retrieve variable domain dictionary from the first factor.
    domains_dict = list(factors)[0].variableDomainsDict()

    # Populate sets with unconditioned and conditioned variables from all factors.
    for factor in factors:
        unconditioned_variables |= set(factor.unconditionedVariables())
        conditioned_variables |= set(factor.conditionedVariables())

    # Remove any conditioned variables that are also unconditioned.
    conditioned_variables -= unconditioned_variables

    # Instantiate a new factor with no overlapping variables.
    crafted_factor = Factor(unconditioned_variables, conditioned_variables, domains_dict)

    # Assign probabilities to each possible combination of variable assignments.
    for assignment in crafted_factor.getAllPossibleAssignmentDicts():
        # Start with a neutral probability multiplier.
        probability_multiplier = 1
        # Combine probabilities across all factors for this assignment.
        for factor in factors:
            probability_multiplier *= factor.getProbability(assignment)
        # Set the aggregated probability for the current assignment.
        crafted_factor.setProbability(assignment, probability_multiplier)
    # The function concludes by returning the composed factor.
    return crafted_factor


def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
        """
        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        "*** YOUR CODE HERE ***"
        # Define variables from the current factor, excluding the one to be eliminated.
        active_unconditioned_vars = factor.unconditionedVariables() - {eliminationVariable}
        dependent_vars = factor.conditionedVariables()
        domains = factor.variableDomainsDict()

        # Formulate a new factor without the eliminated variable.
        crafted_factor = Factor(active_unconditioned_vars, dependent_vars, domains)

        # Calculate probabilities for the new factor over all viable assignments.
        for current_assignment in crafted_factor.getAllPossibleAssignmentDicts():
            # Start with zero and incrementally add to this probability.
            aggregated_probability = 0

            # Iterate over possible values of the eliminated variable and sum their probabilities.
            for elimination_var_value in domains[eliminationVariable]:
                # Create a new assignment dictionary for the current scenario.
                scenario_assignment = {**current_assignment, eliminationVariable: elimination_var_value}
                # Increase the aggregated probability by the current scenario's probability.
                aggregated_probability += factor.getProbability(scenario_assignment)

            # Assign the total aggregated probability to the current assignment in the new factor.
            crafted_factor.setProbability(current_assignment, aggregated_probability)

        # Output the revised factor with updated probability values.
        return crafted_factor

    
    return eliminate

eliminate = eliminateWithCallTracking()

