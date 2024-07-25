# Generic Network Time Series Data Generator

## Variants

### Graph Type

#### Static Graphs:
Edges don't change over time

#### Dynamic Graphs:
Edges change over time

### Node Type

#### Homogenous
Number of node features remain same for all nodes

#### Heterogenous
Number of node features is not constant for all nodes

## Structure

### Node Data

* timestamp: Can be used as integer indices or converted to timestamp in notebooks later
* node: Node ID
* feature: Node Feature ID
* value: Generated value in the range 0 to 1

### Edge Data

* timestamp: Can be used as integer indices or converted to timestamp in notebooks later
* source: Source Node ID
* target: Target Node ID
* feature: Edge Feature ID
* value: Edge feature value

Note: This is exhaustive and pairs of source target not containing an edge has None as value

## Progress

✅ Static - Homogenous 