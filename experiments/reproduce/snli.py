import os

from reproduce.snli_jobs import jobs

if __name__ == "__main__":
    # change current directory to InferSent
    os.chdir('./InferSent/')
    for i, job in enumerate(jobs):
        print("\nJob {} out of {}".format(i + 1, len(jobs)))
        os.system(job)
    # change current directory back to original
    os.chdir('..')
