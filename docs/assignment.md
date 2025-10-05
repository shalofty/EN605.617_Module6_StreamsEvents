# Module 06 - CUDA Streams and Events Assignment

**Due:** Sunday by 11:59pm  
**Points:** 100  
**Submitting:** a text entry box or a file upload  
**Attempts:** 0 Allowed Attempts 3  
**Available:** Sep 15 at 12am - Oct 12 at 11:59pm

## Instructions

1. Download the attached [CUDA_Streams_and_Events_assignment.pdf](CUDA_Streams_and_Events/docs/CUDA_Streams_and_Events_assignment.pdf)

2. Create program that utilizes CUDA streams and events. Any comparison of timing, if the kernel code is the same, will earn extra points.

3. You will need to include the zipped up code for the assignment and images/video/links that show your code completing all of the parts of the rubric that it is designed to complete in what is submitted for this assignment.

> **Note:** Up to 5% of extra credit can be earned if the assignment submission can be demonstrated (via submission text) to be part of your final project's code.

## Scoring

This assignment is worth **100 points**.

## Rubric

### CUDA Streams and Events Assignment Rubric

| Criteria | Ratings | Points |
|----------|---------|--------|
| **Create a program that shows usage of CUDA Streams and Events** | | |
| | 50 pts - **Proficient** - Utilize either streams and events | |
| | 25 pts - **Competent** - Utilize either streams or events | |
| | 0 pts - **Novice** | |
| | | **50 pts** |
| **Test harness executes at least 2 separate runs of each distinct part of your program** | | |
| | 10 pts - **Proficient** - Executes two runs | |
| | 5 pts - **Competent** - Only executes one run | |
| | 0 pts - **Novice** | |
| | | **10 pts** |
| **Output timing or other metrics for comparison of different numbers of threads and block sizes** | | |
| | 10 pts - **Proficient** - metrics compared in an accurate manner | |
| | 5 pts - **Competent** - minimal metrics output | |
| | 0 pts - **Novice** | |
| | | **10 pts** |
| **Quality of your code, measured by use of constants, well-named variables and functions, and useful comments in code** | | |
| | 20 pts - **Proficient** - High quality code | |
| | 10 pts - **Competent** - Good quality code | |
| | 0 pts - **Novice** - decent, good, average quality code that can generally be read/maintained | |
| | | **20 pts** |
| **Command Line tool accepts input of number of threads and block sizes** | | |
| | 10 pts - **Proficient** - executable can be run with both different number of threads and block sizes, configurable by command line arguments | |
| | 5 pts - **Competent** - executable can be run with either different number of threads or block sizes, configurable by command line arguments | |
| | 0 pts - **Novice** | |
| | | **10 pts** |

**Total Points: 100**

## Additional Requirements from PDF

### Detailed Instructions

Create a program that demonstrates understanding of CUDA Streams and Events. It should perform non-trivial operations, such as performing a variety of mathematical calculations or something even more elaborate. There should be a demonstrable use of streams and events and if timing is measured and presented there will be more credit given to that. Part of your submission should include screen capture of output or something similar, showing your code running with a variety of threads and block sizes.

For your assignment submission, you will need to include either a link to the commit/branch for your assignment submission (preferred method), including all code and artifacts, or the zipped up code for the assignment and images/video/links that show your code completing all of the parts of the rubric that it is designed to complete in what is submitted for this assignment.

If you can show how this will be utilized in your final project, you will get a one-time bonus of 5%.

### Task Breakdown

| Task | % of Grade |
|------|------------|
| Create a program that shows usage of CUDA Streams and Events | 50% |
| Test harness executes two separate kernels (or two executions of the same kernel) using CUDA streams and events | 10% |
| Output timing or other metrics for comparison of different datasets | 10% |
| Quality of your code, measured by use of constants, well-named variables and functions, and useful comments in code | 20% |
| Command Line tool accepts input of number of threads and block sizes | 10% |
| **One-time bonus:** Your submission will be part of your final project | 5% |

### Key Requirements Summary

- **Non-trivial operations**: Mathematical calculations or more elaborate tasks
- **Demonstrable streams & events usage**: Clear parallel execution with synchronization
- **Timing measurements**: Critical for extra credit
- **Two separate kernel executions**: Test harness requirement
- **Different datasets**: For timing comparison
- **Command line configurable**: Thread and block sizes
- **Screen captures**: Show code running with various configurations