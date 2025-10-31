# Evolutionary Rankyng System

This is the repo for the final project of the "Optimization for AI" course.

The goal of this project is to create a novable optimization method for zero-sum games (and for all problems that can be map into a zero-sum game).\\
The main idea is to use the already existing ELO system (based not on a in-game fitness function but on a statistical off-game fitness function) in order to generate individuals capable of playing always better and better (until the reach the global optimum that we could call "meta" in this scenario).\\
In addiction this system add some extra components by evolutionary strategies, incorporating mutations and crossovers, in order to leverage the exploration (the ELO system should works as a leverage for the exploitation).\\
In the end propers ranks will be added in order to create different "level of difficulty" (such as bronze, silver, gold etc) in order to make the progression for new individual more gradual (since each rank should be increasingly more difficult to reach since individuals populating a rank should be stronger and stronger).

---

To begin with we must talk about what is the ELO.

## ELO

> The Elo (ranking) system is a method for calculating the relative skill levels of players in zero-sum games such as chess or esports. (...) The difference in the ratings between two players serves as a predictor of the outcome of a match. Two players with equal ratings who play against each other are expected to score an equal number of wins. A player whose rating is 100 points greater than their opponent's is expected to score 64%; if the difference is 200 points, then the expected score for the stronger player is 76%.

from [*Wikipedia*](https://en.wikipedia.org/wiki/Elo_rating_system)

Whenever an individual plays against an other one it can gain or lose ELO points based on the match's result and in the prior success' probability.

The ELO ranking system is something that must be tuned and it is not a really easy task.\\
For that, during this project, I will use a simple version of it defined as follows:

1. Starting with a population of $n$ individuals, each of them will have an uniform probability of winning against each other. This means that $ELO(x) = c \forall x \in Population$, where $c \in \mathbb{N}$ is a positive integer.

2. When competing individuals will gain and lose point based on the relative probability of winning/loosing. This probability will be something proportional to ratio of the ELO of two individuals. The actual proposal for this function will be: $Gain(x, y) = \alpha \frac{ELO(y)}{ELO(x)}$. Note that the Loss function is the same just with the inverted sign.

3. At the end of the match update the ELO of each individual based on the actual GAIN or LOSS.

---

Now let's talk about individuals.

## INDIVIDUALS

Individuals should be simple yet effective algorithms capable of learning from the matches (a really important thing about this whole system is to help moving in the learning space).\\
The idea for this project is to let individuals be ????.

### MUTATIONS and CROSSOVERS

> Here will be described how mutations and crossovers work in this system.

---

## RANKS

> Here will be presented the idea for the ranks (bronze silver gold etc).

---

## PLACEMENTS MATCHES and MUTATIONS

> Here will be presented the mechanism of "placements matches" and how it affects crossovers and mutations (if the offspring beat or not the parent).