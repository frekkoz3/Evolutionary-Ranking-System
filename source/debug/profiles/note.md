# PROFILES

This folder contains the **profiles obtained by cProfile** of the entire system during different steps of the implementation.\
For each profile a little note will be uploaded to help grasp what could be done to improve the efficiency.

1. **📄 profile.prof 📅 24.11.2025 ⏱️ 16.58**: this profile is produced running *main.py with the parallel flag set to true* (using joblib). It is important to notice that joblib is actually improving performances but there is a big overhead driven by all the copy and deep copy. I tried to switch to Ray using the shared memory but since it is (indeed) immutable, there are no real big advantages. To avoid all those operations one could think at changing the player class implementation such that each attribute can actually be stored in a real shared memory.

---

To visualize the desired profiles please use the command:

> snakeviz "name_of_the_desired_profile.prof"
