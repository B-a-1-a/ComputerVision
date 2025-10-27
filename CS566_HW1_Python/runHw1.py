# runHw1.py
# runHw1 is the "main" interface that lets you execute all the
# walkthroughs and challenges in homework 1. It lists a set of
# functions corresponding to the problems that need to be solved.
#
# Note that this file also serves as the specifications for the functions
# you are asked to implement. In some cases, your submissions will be autograded.
# Thus, it is critical that you adhere to all the specified function signatures.
#
# Before your submission, make sure you can run runHw1('all')
# without any error.
#
# Usage:
# python runHw1.py                     : list all the registered functions
# python runHw1.py 'method name'       : execute a specific test
# python runHw1.py all                 : execute all the registered functions

import sys
from signAcademicPolicy import sign_academic_honesty_policy
from hw1_walkthrough1 import hw1_walkthrough1
from hw1_walkthrough2 import hw1_walkthrough2


def runHw1(*args):
    fun_handles = {
        "honesty": honesty,
        "walkthrough1": walkthrough1,
        "walkthrough2": walkthrough2,
    }
    runTests(args, fun_handles)


def honesty():
    # Type your full name and uni (both in string) to state your agreement
    # to the Code of Academic Integrity.
    sign_academic_honesty_policy("Bala Shukla", "shukla35")


def walkthrough1():
    # Open hw1_walkthrough1.py and go through a short Python tutorial
    # You are not required to submit any code for this Walkthrough 1.
    hw1_walkthrough1()


def walkthrough2():
    # Fill in the partially complete code in hw1_walkthrough2.py.
    # Submit the completed code and the outputs
    hw1_walkthrough2()


def runTests(args, fun_handles):
    if not args:
        print("Registered functions:")
        for f in fun_handles:
            print(" -", f)
        return
    arg = args[0]
    if arg == "all":
        for name, func in fun_handles.items():
            print(f"Running {name}()...")
            func()
    elif arg in fun_handles:
        print(f"Running {arg}()...")
        fun_handles[arg]()
    else:
        print("Unknown function name:", arg)


# Allow running from command line
if __name__ == "__main__":
    runHw1(*sys.argv[1:])
