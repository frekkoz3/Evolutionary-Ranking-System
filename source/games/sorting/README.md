# SORTING

This environment is a mapping of the *sorting networks against arrays* problem.\
This is a simple problem where there are two types of individuals which evolves during time.\
The first type of individual is a **sorting network** while the second is an **array**.\
While there are no need to explain what an array is, we spend some time talking about sorting networks.

---

## SORTING NETWORKS

> In computer science, comparator networks are abstract devices built up of a fixed number of "wires", carrying values, and comparator modules that connect pairs of wires, swapping the values on the wires if they are not in a desired order. Such networks are typically designed to perform sorting on fixed numbers of values, in which case they are called sorting networks.

from [*Wikipedia*](https://en.wikipedia.org/wiki/Sorting_network)

Since sorting networks works only with a fixed size, it can be possible to build a larger sorting network and then to pad smaller arrays to sort them with the large sorting network.

An other interesting thing is that sorting network can be easily implemented in hardware and they are easily parallelizable.

---

## SORTING BATTLE
