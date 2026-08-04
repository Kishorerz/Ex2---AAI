<H3>Enter Name :  KISHOR KUMAR B </H3>
<H3>Enter Register No: 212223240072 </H3>
<H3>Experiment 2</H3>
<H3>Date : 4.8.2026</H3>
<h1 align =center>Implementation of Exact Inference Method of Bayesian Network</h1>

## Aim:
To implement the inference Burglary P(B| j,⥗m) in alarm problem by using Variable Elimination method in Python.

## Algorithm:

Step 1: Define the Bayesian Network structure for alarm problem with 5 random variables, Burglary,Earthquake,John Call,Mary Call and Alarm.<br>
Step 2: Define the Conditional Probability Distributions (CPDs) for each variable using the TabularCPD class from the pgmpy library.<br>
Step 3: Add the CPDs to the network.<br>
Step 4: Initialize the inference engine using the VariableElimination class from the pgmpy library.<br>
Step 5: Define the evidence (observed variables) and query variables.<br>
Step 6: Perform exact inference using the defined evidence and query variables.<br>
Step 7: Print the results.<br>

## Program :
```

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

network = DiscreteBayesianNetwork([
    ('Burglary', 'Alarm'),
    ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'),
    ('Alarm', 'MaryCalls')
])

cpd_burglary = TabularCPD('Burglary', 2, [[0.999], [0.001]])
cpd_earthquake = TabularCPD('Earthquake', 2, [[0.998], [0.002]])

cpd_alarm = TabularCPD(
    'Alarm',
    2,
    [
        [0.999, 0.71, 0.06, 0.05],
        [0.001, 0.29, 0.94, 0.95]
    ],
    evidence=['Burglary', 'Earthquake'],
    evidence_card=[2, 2]
)

cpd_john = TabularCPD(
    'JohnCalls', 2,
    [[0.95, 0.1],
     [0.05, 0.9]],
    evidence=['Alarm'],
    evidence_card=[2]
)

cpd_mary = TabularCPD(
    'MaryCalls', 2,
    [[0.99, 0.3],
     [0.01, 0.7]],
    evidence=['Alarm'],
    evidence_card=[2]
)

network.add_cpds(cpd_burglary, cpd_earthquake, cpd_alarm, cpd_john, cpd_mary)

inference = VariableElimination(network)

evidence = {'JohnCalls': 1, 'MaryCalls': 0}
result = inference.query(variables=['Burglary'], evidence=evidence)
print(result)

evidence1 = {'JohnCalls': 1, 'MaryCalls': 1}
result2 = inference.query(variables=['Burglary'], evidence=evidence1)
print(result2)

```

## Output :
<img width="473" height="388" alt="590833082-65b122b9-2a71-4f95-b6e3-1e5f530f0f6a" src="https://github.com/user-attachments/assets/a65acc43-ecd4-4cbd-afd4-03e2bc3c2773" />

## Result :
Thus, Bayesian Inference was successfully determined using Variable Elimination Method

